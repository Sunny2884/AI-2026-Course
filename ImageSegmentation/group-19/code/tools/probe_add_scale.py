"""Add one 1.5-scale global forward per image to a frozen 32-scene tiling probe."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.postprocess import decode_records
from m2f_rgbd.tiling import candidate_recall, map_tile_record, mask_nms
from m2f_rgbd.utils import project_path, write_json
from predict_m2f import load_model_m2f


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def by_image(records):
    result = defaultdict(list)
    for record in records:
        result[record["image_id"]].append(record)
    return result


def box_nms(records):
    if not records:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in
                          [r["bbox"] for r in records]], dtype=torch.float32)
    scores = torch.tensor([r["score"] for r in records], dtype=torch.float32)
    return [records[i] for i in nms(boxes, scores, .7)[:100].tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, out = Path(args.base).resolve(), Path(args.output).resolve()
    frozen = json.loads((base / "manifest.json").read_text())
    assert len(frozen["images"]) == 32, "This experiment adds only 32 new forwards"
    assert digest(frozen["checkpoint"]) == frozen["checkpoint_sha256"], "Weights changed"
    for source, expected in frozen["source_sha256"].items():
        assert digest(project_path(source)) == expected, "Source changed: " + source
    out.mkdir(exist_ok=False)
    cache = out / "views"; cache.mkdir()
    manifest = {"base_manifest": str(base / "manifest.json"),
                "checkpoint_sha256": frozen["checkpoint_sha256"], "image_ids": frozen["images"],
                "added_global_size": 1536, "added_forwards_cap": 32,
                "precision": frozen["precision"], "score_mode": "class_fp32",
                "mask_nms_iou": .5, "maxDets": [1, 10, 100],
                "script_sha256": digest(__file__), "runtime_cap_seconds": 300}
    write_json(out / "manifest.json", manifest)
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    model, config, device = load_model_m2f(frozen["checkpoint"], "cuda")
    torch.cuda.set_per_process_memory_fraction(.45)
    ds = RGBDInstanceDataset(project_path(config["data_root"]), "val", mode=config["mode"],
                             jitter=None, crop_size=None)
    index = {image_id: i for i, image_id in enumerate(ds.ids)}
    truth = ds.build_coco(); annotations = defaultdict(list)
    for ann in truth.dataset["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations[ann["image_id"]].append(ann)
    previous = by_image(read_gzip(base / "global_plus_tiles_mask_nms.json.gz"))
    names = ["s125_plus_tiles", "s150_plus_tiles", "s125_s150_plus_tiles", "s150_global_box_nms"]
    predictions = {name: [] for name in names}
    recalls = {name: {group: defaultdict(int) for group in ["small", "tiny"]} for name in names}
    oracle_names = ["s125_tiles_candidates", "s125_s150_tiles_candidates"]
    oracles = {name: {group: defaultdict(int) for group in ["small", "tiny"]} for name in oracle_names}
    started = time.monotonic(); added_times = []

    def accumulate(target, statistics):
        for group, values in statistics.items():
            for key, value in values.items():
                target[group][key] += value

    with torch.inference_mode():
        model.eval()
        for number, image_id in enumerate(frozen["images"], 1):
            assert time.monotonic() - started < 300, "Probe runtime cap exceeded"
            item = ds[index[image_id]]
            values = F.interpolate(item["pixel_values"][None].to(device), size=(1536, 1536),
                                   mode="bilinear", align_corners=False)
            valid = F.interpolate(item["pixel_mask"][None, None].to(device).float(), size=(1536, 1536),
                                  mode="nearest")[:, 0].bool()
            torch.cuda.synchronize(); begin = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(pixel_values=values, pixel_mask=valid)
            s150 = decode_records(output, [image_id], score_mode="class_fp32")
            torch.cuda.synchronize(); added_times.append(time.monotonic() - begin)
            with gzip.open(cache / f"{image_id}_global150.json.gz", "wt") as stream:
                json.dump(s150, stream, allow_nan=False)
            s125 = read_gzip(base / "views" / f"{image_id}_global125.json.gz")
            tiles = []
            for y in frozen["tile_positions"]:
                for x in frozen["tile_positions"]:
                    local = read_gzip(base / "views" / f"{image_id}_tile_{x}_{y}.json.gz")
                    for record in local:
                        mapped = map_tile_record(record, x, y, frozen["tile_original_size"])
                        if mapped is not None:
                            tiles.append(mapped)
            reproduced = mask_nms(s125 + tiles)
            assert reproduced == previous[image_id], "Frozen tiling baseline changed"
            variants = {"s125_plus_tiles": reproduced,
                        "s150_plus_tiles": mask_nms(s150 + tiles),
                        "s125_s150_plus_tiles": mask_nms(s125 + s150 + tiles),
                        "s150_global_box_nms": box_nms(s150)}
            for name, records in variants.items():
                predictions[name].extend(records)
                accumulate(recalls[name], candidate_recall(records, annotations[image_id]))
            accumulate(oracles[oracle_names[0]], candidate_recall(s125 + tiles, annotations[image_id]))
            accumulate(oracles[oracle_names[1]], candidate_recall(s125 + s150 + tiles, annotations[image_id]))
            print(f"Completed {number}/32, added forwards={number}, elapsed={time.monotonic()-started:.1f}s", flush=True)
            del output, values, valid, item
    del model; torch.cuda.empty_cache()
    metrics = {}
    for name in names:
        metrics[name] = evaluate_predictions(truth, predictions[name], frozen["images"])
        with gzip.open(out / f"{name}.json.gz", "wt") as stream:
            json.dump(predictions[name], stream, allow_nan=False)

    def normalized(source):
        return {name: {group: {**values, **{key.replace("matched_", "recall_"): value / max(values["gt"], 1)
                                            for key, value in values.items() if key.startswith("matched_")}}
                       for group, values in groups.items()} for name, groups in source.items()}
    old = metrics["s125_plus_tiles"]["segm"]["AP_s"]
    new = metrics["s125_s150_plus_tiles"]["segm"]["AP_s"]
    summary = {"images": 32, "new_forwards": 32, "metrics": metrics,
               "recall": normalized(recalls), "candidate_recall": normalized(oracles),
               "additional_scale_AP_s_gain": new - old,
               "additional_scale_relative_gain": new / old - 1,
               "mean_added_forward_and_decode_seconds": sum(added_times) / len(added_times),
               "elapsed_seconds": time.monotonic() - started,
               "note": "Same frozen scenes, weights, score mode and merge thresholds; no training or test submission"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("ADDITIONAL_SCALE_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
