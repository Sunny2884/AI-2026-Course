"""Compare scoring on cached logits; select on val, then export test once."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.postprocess import SCORE_MODES, decode_records, decode_record_variants
from m2f_rgbd.utils import project_path, write_json
from predict import export_submission
from predict_m2f import load_model_m2f


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    names = ["m2f_rgbd/model.py", "m2f_rgbd/data.py", "m2f_rgbd/postprocess.py",
             "tools/compare_postprocess.py"]
    return {name: sha256(project_path(name)) for name in names}


def score_statistics(records):
    counts = Counter(record["score"] for record in records)
    n = len(records)
    return {"instances": n, "unique_scores": len(counts),
            "score_ones": counts[1.0],
            "largest_tie_fraction": max(counts.values(), default=0) / max(n, 1)}


def pilot_indices(dataset, seed):
    groups = defaultdict(list)
    for index, info in enumerate(dataset.images):
        scene = str(info.get("scene_id", Path(info["file_name"]).stem.split("_v")[0]))
        groups[scene].append(index)
    rng = random.Random(seed)
    # One random view per scene; all validation scenes receive equal exposure.
    return sorted(rng.choice(groups[scene]) for scene in sorted(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["pilot", "full", "test"], required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = project_path(args.checkpoint)
    torch.set_num_threads(2)
    checkpoint_hash = sha256(checkpoint)
    hashes = source_hashes()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["checkpoint_sha256"] == checkpoint_hash, "Checkpoint changed"
        assert manifest["source_hashes"] == hashes, "Source changed; use a new output/cache"
    else:
        assert args.phase == "pilot", "Start with the pilot phase"
        manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "source_hashes": hashes, "seed": args.seed,
                    "forward_precision": "bfloat16", "input_size": 1024,
                    "mask_threshold": 0.5, "score_threshold": 0.001,
                    "tta": False, "maxDets": [1, 10, 100]}
        write_json(manifest_path, manifest)
    done = out / (args.phase + "_summary.json")
    if done.exists():
        print("Already complete:", done, flush=True)
        return
    split = "test" if args.phase == "test" else "val"
    if args.phase != "pilot":
        assert (out / "pilot_summary.json").exists(), "Pilot not complete"
    if args.phase == "test":
        full = json.loads((out / "full_summary.json").read_text())
        modes = [full["winner"]]
    else:
        modes = list(SCORE_MODES)
    # Load model once. No optimizer, gradient, or checkpoint writes.
    model, config, device = load_model_m2f(checkpoint, "cuda")
    assert device.type == "cuda"
    assert torch.cuda.device_count() == 1, "Explicitly expose exactly one GPU"
    torch.cuda.set_per_process_memory_fraction(0.60)
    dataset = RGBDInstanceDataset(project_path(config["data_root"]), split,
                                  mode=config["mode"], jitter=None, crop_size=None)
    indices = pilot_indices(dataset, args.seed) if args.phase == "pilot" else list(range(len(dataset)))
    if args.phase == "pilot":
        manifest["pilot_image_ids"] = [dataset.ids[index] for index in indices]
        write_json(manifest_path, manifest)
    decoded_dir = out / ("decoded_" + split)
    raw_dir = out / "pilot_logits"
    decoded_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    records = {mode: [] for mode in modes}
    started = time.monotonic()
    forwards = 0
    with torch.inference_mode():
        model.eval()
        for number, index in enumerate(indices, 1):
            image_id = dataset.ids[index]
            decoded_path = decoded_dir / f"{image_id}.json.gz"
            if decoded_path.exists():
                with gzip.open(decoded_path, "rt") as stream:
                    decoded = json.load(stream)
                assert all(mode in decoded for mode in modes), "Incompatible decoded cache"
            else:
                cache = raw_dir / f"{image_id}.pt"
                if split == "val" and cache.exists():
                    tensors = torch.load(cache, map_location=device, weights_only=True)
                else:
                    item = dataset[index]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(pixel_values=item["pixel_values"][None].to(device),
                                           pixel_mask=item["pixel_mask"][None].to(device))
                    tensors = {"class_queries_logits": prediction.class_queries_logits.detach(),
                               "masks_queries_logits": prediction.masks_queries_logits.detach()}
                    forwards += 1
                    if args.phase == "pilot":
                        temporary = cache.with_suffix(".tmp")
                        torch.save({key: value.cpu() for key, value in tensors.items()}, temporary)
                        temporary.replace(cache)
                    del prediction, item
                outputs = SimpleNamespace(**tensors)
                if set(modes) == set(SCORE_MODES):
                    decoded = decode_record_variants(outputs, [image_id])
                else:
                    decoded = {mode: decode_records(outputs, [image_id], score_mode=mode)
                               for mode in modes}
                temporary = decoded_path.with_suffix(".tmp")
                with gzip.open(temporary, "wt") as stream:
                    json.dump(decoded, stream, allow_nan=False)
                temporary.replace(decoded_path)
                del outputs, tensors
            for mode in modes:
                records[mode].extend(decoded[mode])
            if number == 1 or number % 25 == 0 or number == len(indices):
                print(f"{args.phase}: {number}/{len(indices)}, forwards={forwards}, "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    image_ids = [dataset.ids[index] for index in indices]
    result = {"phase": args.phase, "images": len(indices), "new_forwards": forwards,
              "checkpoint_sha256": checkpoint_hash, "score_statistics": {}, "results": {}}
    if split == "val":
        truth = dataset.build_coco()
        for mode in modes:
            metrics_path = out / f"{args.phase}_{mode}_metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
            else:
                metrics = evaluate_predictions(truth, records[mode], image_ids)
                write_json(metrics_path, metrics)
            result["results"][mode] = metrics
            result["score_statistics"][mode] = score_statistics(records[mode])
            print(mode, "AP_s=", metrics["segm"]["AP_s"], "AP=", metrics["segm"]["AP"], flush=True)
        winner = max(modes, key=lambda mode: result["results"][mode]["segm"]["AP_s"])
        result["winner"] = winner
        result["gain_over_legacy"] = (result["results"][winner]["segm"]["AP_s"] -
                                      result["results"]["legacy_bf16"]["segm"]["AP_s"])
        result["gain_over_class_fp32"] = (result["results"][winner]["segm"]["AP_s"] -
                                          result["results"]["class_fp32"]["segm"]["AP_s"])
    else:
        mode = modes[0]
        submission_dir = out / ("submission_" + mode)
        assert not submission_dir.exists(), "Refusing to overwrite submission directory"
        submission = export_submission(records[mode], dataset.ids,
                                       project_path(config["data_root"]) / "submission_3.csv",
                                       submission_dir, csv_format="masks", partial=False)
        submission.update(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                          score_mode=mode, forward_precision="bfloat16",
                          postprocess_precision="bfloat16" if mode == "legacy_bf16" else "float32",
                          score_threshold=0.001, mask_threshold=0.5, tta=None,
                          model=config["checkpoint_dir"], mode=config["mode"],
                          selection_source=str(out / "full_summary.json"))
        write_json(submission_dir / "submission_info.json", submission)
        validate_submission(submission_dir, dataset.ids)
        result["submission_dir"] = str(submission_dir)
        result["winner"] = mode
        result["score_statistics"][mode] = score_statistics(records[mode])
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(done, result)
    print(json.dumps(result, indent=2), flush=True)
    print("PHASE_COMPLETE", args.phase, flush=True)


def validate_submission(directory, expected_ids):
    from pycocotools import mask as mu
    with open(directory / "submission.csv", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["image_id", "PredictionString"]
        rows = list(reader)
    ids = [int(row["image_id"]) for row in rows]
    assert len(ids) == len(set(ids)) == len(expected_ids)
    assert set(ids) == set(expected_ids)
    total = 0
    for row in rows:
        objects = json.loads(row["PredictionString"])
        assert isinstance(objects, list) and len(objects) <= 100
        for obj in objects:
            assert set(obj) == {"category_id", "score", "segmentation"}
            assert obj["category_id"] == 1 and np.isfinite(obj["score"]) and 0 <= obj["score"] <= 1
            mask = obj["segmentation"]
            assert mask["size"] == [1024, 1024] and isinstance(mask["counts"], str)
            assert mu.area(mask) > 0
        total += len(objects)
    exported = json.loads((directory / "submission.json").read_text())
    assert len(exported) == total
    print("CSV/RLE validation passed:", len(ids), "images,", total, "instances", flush=True)


if __name__ == "__main__":
    main()
