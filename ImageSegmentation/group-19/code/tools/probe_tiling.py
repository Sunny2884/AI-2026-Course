"""Bounded, no-training probe: global 1.25 vs nine enlarged overlapping tiles."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from pycocotools import mask as mu
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.postprocess import decode_records
from m2f_rgbd.tiling import candidate_recall, map_tile_record, mask_nms, tile_positions
from m2f_rgbd.utils import project_path, write_json
from predict_m2f import load_model_m2f


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def select_indices(dataset, annotations, count, seed):
    groups = defaultdict(list)
    tiny_counts = defaultdict(int)
    for image_id, anns in annotations.items():
        tiny_counts[image_id] = sum(0 < ann["area"] < 256 for ann in anns)
    for index, info in enumerate(dataset.images):
        scene = str(info.get("scene_id", Path(info["file_name"]).stem.split("_v")[0]))
        groups[scene].append(index)
    scenes = sorted(groups, key=lambda scene: (np.mean([tiny_counts[dataset.ids[i]]
                                                       for i in groups[scene]]), scene))
    rng = random.Random(seed)
    selected = []
    for stratum, partition in enumerate(np.array_split(np.array(scenes), 4)):
        n = count // 4 + int(stratum < count % 4)
        for scene in rng.sample(partition.tolist(), n):
            selected.append(rng.choice(groups[scene]))
    return sorted(selected)


def box_nms(records):
    from torchvision.ops import nms
    if not records:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in
                          [r["bbox"] for r in records]], dtype=torch.float32)
    scores = torch.tensor([r["score"] for r in records], dtype=torch.float32)
    return [records[i] for i in nms(boxes, scores, .7)[:100].tolist()]


def sum_recall(total, addition):
    for group, values in addition.items():
        for key, value in values.items():
            total[group][key] += value


def normalize_recall(stats):
    return {group: {**values, **{key.replace("matched_", "recall_"): value / max(values["gt"], 1)
                                for key, value in values.items() if key.startswith("matched_")}}
            for group, values in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if not 4 <= args.images <= 32:
        parser.error("This probe is capped at 32 images")
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    assert not (out / "summary.json").exists(), "Probe already complete"
    torch.set_num_threads(2)
    checkpoint = project_path(args.checkpoint)
    model, config, device = load_model_m2f(checkpoint, "cuda")
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(.45)
    dataset = RGBDInstanceDataset(project_path(config["data_root"]), "val", mode=config["mode"],
                                  jitter=None, crop_size=None)
    truth = dataset.build_coco()
    annotations = defaultdict(list)
    for ann in truth.dataset["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations[int(ann["image_id"])].append(ann)
    indices = select_indices(dataset, annotations, args.images, 20261025)
    tile_size, stride = 448, 288
    positions = tile_positions(1024, tile_size, stride)
    assert len(positions) ** 2 == 9
    source_names = ["m2f_rgbd/model.py", "m2f_rgbd/data.py", "m2f_rgbd/postprocess.py",
                    "m2f_rgbd/tiling.py", "tools/probe_tiling.py", "predict_m2f.py"]
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "source_sha256": {name: digest(project_path(name)) for name in source_names},
                "images": [dataset.ids[i] for i in indices], "seed": 20261025,
                "sampling": "8 scenes per quartile of scene-level tiny-instance count; one random view per scene",
                "global_input": 1280, "tile_original_size": tile_size, "tile_network_input": 1024,
                "tile_stride": stride, "tile_positions": positions, "max_views_per_image": 10,
                "precision": "bf16 forward, fp32 mask interpolation and class score",
                "score_threshold": .001, "mask_threshold": .5, "mask_nms_iou": .5,
                "box_nms_iou": .7, "maxDets": [1, 10, 100], "runtime_cap_seconds": args.seconds,
                "selection_gate": {"AP_s_ratio_minimum": 1.30, "tiny_recall_50_gain_minimum": .15,
                                   "small_recall_75_gain_minimum": .10},
                "note": "Screening experiment, not a test-score estimate or trained tiling method"}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Incompatible cache"
    else:
        write_json(manifest_path, manifest)
    cache = out / "views"; cache.mkdir(exist_ok=True)
    started = time.monotonic(); forwards = 0; complete_ids = []
    names = ["global_box_nms", "global_mask_nms", "global_plus_tiles_mask_nms"]
    predictions = {name: [] for name in names}
    recalls = {name: {g: defaultdict(int) for g in ["small", "tiny"]}
               for name in names + ["global_candidates_oracle", "union_candidates_oracle"]}
    times = []; counts = []; budget_exceeded = False

    def run_view(image_id, pixels, pixel_mask, view_name, network_size, output_size):
        nonlocal forwards
        path = cache / f"{image_id}_{view_name}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as stream:
                return json.load(stream)
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Probe runtime cap reached")
        values = F.interpolate(pixels[None].to(device), size=(network_size, network_size),
                               mode="bilinear", align_corners=False)
        valid = F.interpolate(pixel_mask[None, None].to(device).float(), size=(network_size, network_size),
                              mode="nearest")[:, 0].bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=values, pixel_mask=valid)
        records = decode_records(outputs, [image_id], output_size=output_size, score_mode="class_fp32")
        forwards += 1
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump(records, stream, allow_nan=False)
        temporary.replace(path)
        return records

    with torch.inference_mode():
        model.eval()
        for number, index in enumerate(indices, 1):
            begin = time.monotonic(); image_id = dataset.ids[index]; item = dataset[index]
            try:
                torch.cuda.synchronize(); global_begin = time.monotonic()
                global_records = run_view(image_id, item["pixel_values"], item["pixel_mask"], "global125", 1280, 1024)
                torch.cuda.synchronize(); global_seconds = time.monotonic() - global_begin
                union = list(global_records); dropped = 0; local_count = 0
                for y in positions:
                    for x in positions:
                        local = run_view(image_id, item["pixel_values"][:, y:y + tile_size, x:x + tile_size],
                                         item["pixel_mask"][y:y + tile_size, x:x + tile_size],
                                         f"tile_{x}_{y}", 1024, tile_size)
                        local_count += len(local)
                        for record in local:
                            mapped = map_tile_record(record, x, y, tile_size)
                            if mapped is None:
                                dropped += 1
                            else:
                                union.append(mapped)
            except TimeoutError:
                budget_exceeded = True
                print("Runtime cap reached; report only fully processed images", flush=True)
                break
            variants = {"global_box_nms": box_nms(global_records),
                        "global_mask_nms": mask_nms(global_records),
                        "global_plus_tiles_mask_nms": mask_nms(union)}
            for name, records in variants.items():
                predictions[name].extend(records)
                sum_recall(recalls[name], candidate_recall(records, annotations[image_id]))
            sum_recall(recalls["global_candidates_oracle"], candidate_recall(global_records, annotations[image_id]))
            sum_recall(recalls["union_candidates_oracle"], candidate_recall(union, annotations[image_id]))
            seconds = time.monotonic() - begin
            times.append({"global_seconds": global_seconds, "total_seconds": seconds})
            counts.append({"image_id": image_id, "global": len(global_records), "local": local_count,
                           "edge_dropped": dropped, "union": len(union),
                           "final": len(variants["global_plus_tiles_mask_nms"])})
            complete_ids.append(image_id)
            print(f"Completed {number}/{len(indices)}, views={forwards}, global={len(global_records)}, "
                  f"tile_union={len(union)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
            write_json(out / "progress.json", {"completed": complete_ids, "counts": counts, "timing": times})
    del model; torch.cuda.empty_cache()
    assert complete_ids, "No complete image; inspect cache and logs"
    metrics = {}
    for name in names:
        metrics[name] = evaluate_predictions(truth, predictions[name], complete_ids)
        with gzip.open(out / (name + ".json.gz"), "wt") as stream:
            json.dump(predictions[name], stream, allow_nan=False)
    recall_result = {name: normalize_recall(stats) for name, stats in recalls.items()}
    base = metrics["global_box_nms"]["segm"]["AP_s"]
    tiled = metrics["global_plus_tiles_mask_nms"]["segm"]["AP_s"]
    tiny_gain = (recall_result["global_plus_tiles_mask_nms"]["tiny"]["recall_0.50"] -
                 recall_result["global_box_nms"]["tiny"]["recall_0.50"])
    high_iou_gain = (recall_result["global_plus_tiles_mask_nms"]["small"]["recall_0.75"] -
                    recall_result["global_box_nms"]["small"]["recall_0.75"])
    gate = {"AP_s_ratio": tiled / max(base, 1e-9), "tiny_recall_50_gain": tiny_gain,
            "small_recall_75_gain": high_iou_gain}
    gate["passed"] = (len(complete_ids) == len(indices) and gate["AP_s_ratio"] >= 1.30
                      and tiny_gain >= .15 and high_iou_gain >= .10)
    summary = {"images": len(complete_ids), "requested_images": len(indices), "budget_exceeded": budget_exceeded,
               "new_forwards": forwards, "metrics": metrics, "recall": recall_result, "gate": gate,
               "mean_global_seconds": float(np.mean([t["global_seconds"] for t in times])),
               "mean_pipeline_seconds": float(np.mean([t["total_seconds"] for t in times])),
               "elapsed_seconds": time.monotonic() - started,
               "timing_note": "Includes cache writes and CPU candidate diagnostics; not a deployment benchmark"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("BOUNDED_TILING_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
