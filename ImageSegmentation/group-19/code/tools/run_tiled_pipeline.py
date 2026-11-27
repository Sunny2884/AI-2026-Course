"""Resource-aware sharded validation and submission, using frozen tiles5 inference.

One worker per selected GPU, one image per forward, resumable atomic per-image
caches, exact image-ID validation, and a full/holdout validation gate before test.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pycocotools import mask as mu
from torch.nn import functional as F
from torchvision.ops import nms

from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.postprocess import decode_records
from m2f_rgbd.tiling import map_tile_record, mask_nms
from m2f_rgbd.utils import project_path

TILES = [(0, 0), (576, 0), (288, 288), (0, 576), (576, 576)]
SOURCES = ["tools/run_tiled_pipeline.py", "m2f_rgbd/tiling.py", "m2f_rgbd/postprocess.py",
           "m2f_rgbd/model.py", "m2f_rgbd/data.py", "m2f_rgbd/engine.py", "predict_m2f.py", "predict.py"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_item(path, item):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(item, stream, allow_nan=False)
    temporary.replace(path)


def read_item(path, expected_id, signature):
    with gzip.open(path, "rt") as stream:
        item = json.load(stream)
    if item["image_id"] != expected_id or item["signature"] != signature:
        raise ValueError(f"Cache provenance mismatch: {path}")
    for key in ["global", "tiles5"]:
        records = item[key]
        if len(records) > 100:
            raise ValueError("More than 100 final instances")
        for record in records:
            if record["image_id"] != expected_id or record["category_id"] != 1:
                raise ValueError("Invalid record image/category")
            score = record["score"]
            mask = record["segmentation"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid instance score")
            if mask["size"] != [1024, 1024] or not isinstance(mask["counts"], str) or mu.area(mask) <= 0:
                raise ValueError("Invalid or empty RLE mask")
    return item


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def box_nms(records):
    if not records:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in
                          [record["bbox"] for record in records]], dtype=torch.float32)
    scores = torch.tensor([record["score"] for record in records], dtype=torch.float32)
    return [records[i] for i in nms(boxes, scores, .7)[:100].tolist()]


@torch.inference_mode()
def predict_one(model, item, image_id, device):
    def view(pixels, valid, size, original_size):
        values = F.interpolate(pixels[None].to(device), size=(size, size), mode="bilinear", align_corners=False)
        mask = F.interpolate(valid[None, None].to(device).float(), size=(size, size), mode="nearest")[:, 0].bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(pixel_values=values, pixel_mask=mask)
        return decode_records(output, [image_id], output_size=original_size, score_mode="class_fp32")

    global_records = view(item["pixel_values"], item["pixel_mask"], 1536, 1024)
    union = list(global_records)
    for x, y in TILES:
        local = view(item["pixel_values"][:, y:y + 448, x:x + 448],
                     item["pixel_mask"][y:y + 448, x:x + 448], 1024, 448)
        for record in local:
            mapped = map_tile_record(record, x, y, 448)
            if mapped is not None:
                union.append(mapped)
    return box_nms(global_records), mask_nms(union)


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return [{"index": int(i), "free_mib": int(f), "utilization": int(u)}
            for i, f, u in csv.reader(result.stdout.splitlines())]


def choose_gpus(requested, maximum):
    samples = []
    for number in range(3):
        samples.append(gpu_snapshot())
        if number < 2:
            time.sleep(1)
    stats = []
    for first in samples[0]:
        index = first["index"]
        rows = [next(row for row in sample if row["index"] == index) for sample in samples]
        stats.append({"index": index, "min_free_mib": min(row["free_mib"] for row in rows),
                      "mean_utilization": sum(row["utilization"] for row in rows) / len(rows)})
    if requested != "auto":
        chosen = [int(value) for value in requested.split(",")]
        if len(chosen) != len(set(chosen)) or not set(chosen) <= {row["index"] for row in stats}:
            raise ValueError("Invalid GPU selection")
        if any(row["min_free_mib"] < 12000 for row in stats if row["index"] in chosen):
            raise RuntimeError("Selected GPU has less than 12 GiB free")
    else:
        candidates = [row for row in stats if row["min_free_mib"] >= 12000 and row["mean_utilization"] <= 55]
        if not candidates:
            candidates = [row for row in stats if row["min_free_mib"] >= 16000 and row["mean_utilization"] < 85]
        chosen = [row["index"] for row in sorted(candidates,
                  key=lambda row: (row["mean_utilization"], -row["min_free_mib"]))[:maximum]]
    if not chosen:
        raise RuntimeError("No sufficiently available GPU; existing jobs were not stopped")
    return chosen, stats


def worker(args):
    from predict_m2f import load_model_m2f
    root = Path(args.run_root)
    manifest = json.loads((root / "manifest.json").read_text())
    phase = root / args.phase
    ids = json.loads((phase / "image_ids.json").read_text())[args.rank::args.world]
    needed = [image_id for image_id in ids if not (phase / "items" / f"{image_id}.json.gz").exists()]
    torch.set_num_threads(2)
    status = phase / f"worker_{args.rank}_status.json"
    if not needed:
        write_json(status, {"done": len(ids), "total": len(ids), "new_forwards": 0, "complete": True})
        return
    assert torch.cuda.device_count() == 1, "Each worker must see exactly one GPU"
    torch.cuda.set_per_process_memory_fraction(.25)
    model, config, device = load_model_m2f(manifest["checkpoint"], "cuda")
    dataset = RGBDInstanceDataset(project_path(config["data_root"]),
                                  "test" if args.phase == "test" else "val", mode=config["mode"],
                                  jitter=None, crop_size=None)
    lookup = {image_id: index for index, image_id in enumerate(dataset.ids)}
    started = time.monotonic(); new_forwards = 0; new_images = 0
    for image_id in needed:
        # Only pause our worker when remaining memory becomes critically low.
        free, _ = torch.cuda.mem_get_info()
        pause_started = time.monotonic()
        while free < 3 * 1024 ** 3:
            torch.cuda.empty_cache(); time.sleep(5)
            if time.monotonic() - pause_started > 600:
                raise RuntimeError("GPU memory stayed critically low for 10 minutes")
            free, _ = torch.cuda.mem_get_info()
        item = dataset[lookup[image_id]]; before = time.monotonic()
        global_records, tiled_records = predict_one(model, item, image_id, device)
        write_item(phase / "items" / f"{image_id}.json.gz",
                   {"image_id": image_id, "signature": manifest["signature"],
                    "global": global_records, "tiles5": tiled_records,
                    "new_forwards": 6, "seconds": time.monotonic() - before,
                    "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "source": "fresh_inference"})
        new_images += 1; new_forwards += 6
        write_json(status, {"done": len(ids) - len(needed) + new_images, "total": len(ids),
                           "new_forwards": new_forwards, "elapsed_seconds": time.monotonic() - started,
                           "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "complete": new_images == len(needed)})
        if new_images == 1 or new_images % 25 == 0:
            print(f"{args.phase} worker={args.rank} GPU={os.environ['CUDA_VISIBLE_DEVICES']} "
                  f"new={new_images}/{len(needed)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        del item, global_records, tiled_records


def records_by_image(records):
    result = defaultdict(list)
    for record in records:
        result[record["image_id"]].append(record)
    return result


def initialize(args):
    root = Path(args.run_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    checkpoint = project_path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]; del state
    frozen = json.loads(Path(args.pilot_base, "manifest.json").read_text())
    for source in ["m2f_rgbd/model.py", "m2f_rgbd/data.py", "m2f_rgbd/tiling.py", "m2f_rgbd/postprocess.py"]:
        assert digest(project_path(source)) == frozen["source_sha256"][source], "Frozen inference source changed: " + source
    weights_hash = digest(checkpoint)
    if not getattr(args, "allow_new_weights", False):
        assert weights_hash == frozen["checkpoint_sha256"], "Expected the validated frozen weights"
    spec = {"checkpoint": str(checkpoint), "checkpoint_sha256": weights_hash,
            "config": config, "global_size": 1536, "tile_original_size": 448,
            "tile_network_size": 1024, "tile_positions": TILES,
            "precision": "bf16 forward; fp32 postprocessing", "score_mode": "class_fp32",
            "score_threshold": .001, "mask_threshold": .5, "mask_nms_iou": .5,
            "baseline_box_nms_iou": .7, "maxDets": [1, 10, 100],
            "source_hashes": {name: digest(project_path(name)) for name in SOURCES}}
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec["signature"] = signature
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text())["signature"] == signature, "Run source/config changed; use a new run root"
    else:
        write_json(path, spec)
    return root, spec, frozen


def launch_phase(args, root, manifest, phase_name, ids):
    phase = root / phase_name; phase.mkdir(exist_ok=True)
    (phase / "items").mkdir(exist_ok=True)
    if (phase / "image_ids.json").exists():
        assert json.loads((phase / "image_ids.json").read_text()) == ids
    else:
        write_json(phase / "image_ids.json", ids)
    expected = {f"{image_id}.json.gz" for image_id in ids}
    existing = {path.name for path in (phase / "items").glob("*.json.gz")}
    assert existing <= expected, "Unexpected IDs in phase cache"
    if existing == expected:
        return
    gpu_ids, stats = choose_gpus(args.gpus, args.max_gpus)
    write_json(phase / "gpu_selection.json", {"chosen": gpu_ids, "snapshots": stats,
                                              "per_worker_allocator_cap_GiB": 10})
    print("Selected GPUs:", gpu_ids, "for", phase_name, flush=True)
    processes = []; logs = []
    try:
        for rank, gpu in enumerate(gpu_ids):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            log = (phase / f"worker_{rank}.log").open("ab"); logs.append(log)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--phase", phase_name,
                       "--run-root", str(root), "--rank", str(rank), "--world", str(len(gpu_ids))]
            processes.append(subprocess.Popen(command, env=env, cwd=project_path("."),
                                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        last = 0
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in [None, 0] for process in processes):
                raise RuntimeError(f"Worker failed during {phase_name}; inspect worker logs. Completed items are resumable.")
            now = time.monotonic()
            if now - last >= 15:
                count = len(list((phase / "items").glob("*.json.gz")))
                write_json(root / "status.json", {"phase": phase_name, "completed": count,
                                                   "total": len(ids), "gpus": gpu_ids, "state": "running"})
                print(f"{phase_name}: {count}/{len(ids)} images", flush=True); last = now
            time.sleep(2)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(f"Worker failed in {phase_name}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        for log in logs:
            log.close()
    actual = {path.name for path in (phase / "items").glob("*.json.gz")}
    assert actual == expected, f"Missing or extra {phase_name} images"


def load_phase(root, manifest, phase, ids):
    records = {"global": [], "tiles5": []}
    for image_id in ids:
        item = read_item(root / phase / "items" / f"{image_id}.json.gz", image_id, manifest["signature"])
        for name in records:
            records[name].extend(item[name])
    return records


def check_equivalent(actual, expected):
    assert len(actual) == len(expected), "Instance count differs from frozen reference"
    for left, right in zip(actual, expected):
        assert left["segmentation"] == right["segmentation"] and left["bbox"] == right["bbox"], "Mask geometry differs"
        assert abs(left["score"] - right["score"]) <= 2e-5, "Class score differs"


def smoke_and_seed(args, root, manifest, frozen):
    ids = frozen["images"][:3]
    launch_phase(args, root, manifest, "smoke", ids)
    scale = Path(args.pilot_base) / "add_scale150_01"
    expected_global = records_by_image(read_gzip(scale / "s150_global_box_nms.json.gz"))
    expected_tiles = records_by_image(read_gzip(Path(args.pilot_base) / "tile_count_cpu_01" / "tiles5_corners_center.json.gz"))
    for image_id in ids:
        item = read_item(root / "smoke" / "items" / f"{image_id}.json.gz", image_id, manifest["signature"])
        check_equivalent(item["global"], expected_global[image_id])
        check_equivalent(item["tiles5"], expected_tiles[image_id])
    write_json(root / "smoke" / "summary.json", {"images": len(ids), "passed": True,
                                                  "checks": "multi-GPU worker output matches frozen single-GPU reference"})
    (root / "val" / "items").mkdir(parents=True, exist_ok=True)
    for image_id in frozen["images"]:
        path = root / "val" / "items" / f"{image_id}.json.gz"
        if not path.exists():
            write_item(path, {"image_id": image_id, "signature": manifest["signature"],
                              "global": expected_global[image_id], "tiles5": expected_tiles[image_id],
                              "new_forwards": 0, "source": "frozen_pilot_cache"})
    print("Smoke consistency check passed; 32 pilot images reused", flush=True)


def validate(args, root, manifest, frozen):
    from m2f_rgbd.engine import evaluate_predictions
    dataset = RGBDInstanceDataset(project_path(manifest["config"]["data_root"]), "val",
                                  mode=manifest["config"]["mode"], jitter=None, crop_size=None)
    ids = dataset.ids
    assert len(ids) == 2584 and len(set(ids)) == len(ids)
    launch_phase(args, root, manifest, "val", ids)
    path = root / "val" / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    predictions = load_phase(root, manifest, "val", ids)
    truth = dataset.build_coco()
    heldout_ids = [image_id for image_id in ids if image_id not in set(frozen["images"])]
    subsets = {"full": ids, "independent_of_pilot": heldout_ids}
    results = {}
    for subset, subset_ids in subsets.items():
        allowed = set(subset_ids); results[subset] = {}
        for name in predictions:
            metrics_path = root / "val" / f"{subset}_{name}_metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
            else:
                metrics = evaluate_predictions(truth, [record for record in predictions[name]
                                                       if record["image_id"] in allowed], subset_ids)
                write_json(metrics_path, metrics)
            results[subset][name] = metrics
            print(subset, name, "AP_s=", metrics["segm"]["AP_s"], "AP=", metrics["segm"]["AP"], flush=True)
    full_gain = results["full"]["tiles5"]["segm"]["AP_s"] - results["full"]["global"]["segm"]["AP_s"]
    holdout_gain = (results["independent_of_pilot"]["tiles5"]["segm"]["AP_s"] -
                    results["independent_of_pilot"]["global"]["segm"]["AP_s"])
    ap_change = results["full"]["tiles5"]["segm"]["AP"] - results["full"]["global"]["segm"]["AP"]
    gate = full_gain >= .02 and holdout_gain >= .02 and ap_change >= -.01
    summary = {"images": len(ids), "independent_images": len(heldout_ids), "pilot_images": len(frozen["images"]),
               "results": results, "full_AP_s_gain": full_gain, "independent_AP_s_gain": holdout_gain,
               "full_Mask_AP_change": ap_change, "passed_submission_gate": gate,
               "gate": "Full and independent AP_s gain >=0.02; full Mask AP decline no worse than0.01",
               "checkpoint_sha256": manifest["checkpoint_sha256"]}
    write_json(path, summary)
    return summary


def verify_submission(directory, template, expected_ids):
    csv.field_size_limit(16 * 1024 * 1024)
    with open(template, encoding="utf-8-sig", newline="") as stream:
        template_ids = [int(row["image_id"]) for row in csv.DictReader(stream)]
    with open(directory / "submission.csv", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["image_id", "PredictionString"]
        rows = list(reader)
    ids = [int(row["image_id"]) for row in rows]
    assert ids == template_ids and len(ids) == len(set(ids)) == len(expected_ids)
    assert set(ids) == set(expected_ids)
    expected = records_by_image(json.loads((directory / "submission.json").read_text()))
    count = 0
    for row in rows:
        image_id = int(row["image_id"]); objects = json.loads(row["PredictionString"])
        assert isinstance(objects, list) and len(objects) <= 100
        reference = [{key: record[key] for key in ["category_id", "score", "segmentation"]}
                     for record in expected[image_id]]
        assert objects == reference, "CSV/JSON mismatch"
        for obj in objects:
            assert obj["category_id"] == 1 and math.isfinite(obj["score"]) and 0 <= obj["score"] <= 1
            mask = obj["segmentation"]
            assert mask["size"] == [1024, 1024] and isinstance(mask["counts"], str) and mu.area(mask) > 0
        count += len(objects)
    return {"images": len(ids), "instances": count, "template_order": True, "CSV_JSON_equivalent": True,
            "nonempty_compressed_RLE": True, "csv_sha256": digest(directory / "submission.csv")}


def export_test(args, root, manifest, validation):
    from predict import export_submission
    dataset = RGBDInstanceDataset(project_path(manifest["config"]["data_root"]), "test",
                                  mode=manifest["config"]["mode"], jitter=None, crop_size=None)
    assert len(dataset.ids) == 2112
    launch_phase(args, root, manifest, "test", dataset.ids)
    records = load_phase(root, manifest, "test", dataset.ids)["tiles5"]
    directory = Path(args.submission_dir).resolve()
    template = project_path(manifest["config"]["data_root"]) / "submission_3.csv"
    if directory.exists() and any(directory.iterdir()):
        summary_file = directory / "submission_info.json"
        assert summary_file.exists(), "Refusing to overwrite nonempty output"
        summary = json.loads(summary_file.read_text())
        assert summary.get("run_signature") == manifest["signature"], "Output belongs to a different run"
    else:
        summary = export_submission(records, dataset.ids, template, directory, csv_format="masks", partial=False)
        summary.update(checkpoint=manifest["checkpoint"], checkpoint_sha256=manifest["checkpoint_sha256"],
                       run_signature=manifest["signature"], inference="s150_global_plus_corners_center_5tiles",
                       global_scale=1.5, tile_original_size=448, tile_network_input=1024, tile_positions=TILES,
                       score_mode="class_fp32", forward_precision="bf16", postprocess_precision="fp32",
                       score_threshold=.001, mask_threshold=.5, mask_nms_iou=.5,
                       validation_summary=str(root / "val" / "summary.json"), run_root=str(root))
        write_json(directory / "submission_info.json", summary)
    verification = verify_submission(directory, template, dataset.ids)
    write_json(directory / "verification.json", verification)
    write_json(root / "test" / "summary.json", {"submission_dir": str(directory), "verification": verification})
    print("Submission verified:", directory / "submission.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/m2f_finetune_best.pth")
    parser.add_argument("--pilot-base", default="probes/pilot")
    parser.add_argument("--submission-dir", default="outputs/submission_m2f_s150_tiles5")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--max-gpus", type=int, default=3)
    parser.add_argument("--phase", choices=["all", "smoke", "val", "test"], default="all")
    parser.add_argument("--allow-new-weights", action="store_true",
                        help="Permit a checkpoint whose sha differs from the frozen pilot; keeps the frozen tile layout")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args); return
    torch.set_num_threads(2)
    args.submission_dir = str(project_path(args.submission_dir))
    root = Path(args.run_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        try:
            root, manifest, frozen = initialize(args)
            destination = Path(args.submission_dir)
            if destination.exists() and any(destination.iterdir()):
                info = destination / "submission_info.json"
                assert info.exists() and json.loads(info.read_text()).get("run_signature") == manifest["signature"], "Refusing to overwrite existing submission"
            if args.phase in ["all", "smoke"]:
                smoke_and_seed(args, root, manifest, frozen)
            if args.phase == "smoke":
                return
            if args.phase in ["all", "val"]:
                validation = validate(args, root, manifest, frozen)
            else:
                validation = json.loads((root / "val" / "summary.json").read_text())
            if args.phase == "val":
                return
            if not validation["passed_submission_gate"]:
                write_json(root / "status.json", {"state": "validation_gate_failed", "elapsed_seconds": time.monotonic()-started})
                print("Validation gain insufficient; test inference was not started", flush=True)
                return
            export_test(args, root, manifest, validation)
            write_json(root / "status.json", {"state": "complete", "elapsed_seconds": time.monotonic()-started,
                                              "submission_dir": args.submission_dir})
            print("TILED_VALIDATION_AND_SUBMISSION_COMPLETE", flush=True)
        except Exception:
            write_json(root / "status.json", {"state": "failed", "traceback": traceback.format_exc(),
                                              "elapsed_seconds": time.monotonic()-started})
            raise


if __name__ == "__main__":
    main()
