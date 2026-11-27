"""One-shot probe: tile zoom/layout crossed with score mode, on frozen pilot scenes.

Compares four candidates with identical global 1.5x inference:
- tiles5   (448 px, stride 288, corners+center)  x class_fp32 / mask_quality
- tiles9z  (352 px, stride 336, full coverage)   x class_fp32 / mask_quality
The global-only branch is reported for reference. Every view is forwarded once;
the two score modes are decoded from the same logits (class score vs class x
mask-quality), so the comparison isolates ranking rather than sampling noise.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.postprocess import decode_records
from m2f_rgbd.tiling import candidate_recall, map_tile_record, mask_nms, tile_positions
from m2f_rgbd.utils import project_path, write_json
from predict_m2f import load_model_m2f

MODES = ("class_fp32", "mask_quality")
SCORE_THRESHOLD = 0.001
MASK_THRESHOLD = 0.5
NMS_IOU = 0.5
LAYOUTS = {
    "tiles5_448": {"indices": [0, 2, 4, 6, 8], "tile": 448, "stride": 288},
    "tiles9_352": {"indices": list(range(9)), "tile": 352, "stride": 336},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def select_indices(dataset, annotations, count, seed):
    """Stratified scene sampling identical to the frozen pilot probe."""
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


def box_nms(records, iou_threshold=0.7, max_dets=100):
    if not records:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in
                          [record["bbox"] for record in records]], dtype=torch.float32)
    scores = torch.tensor([record["score"] for record in records], dtype=torch.float32)
    return [records[i] for i in nms(boxes, scores, iou_threshold)[:max_dets].tolist()]


@torch.inference_mode()
def view_variants(model, pixels, valid, network_size, output_size, image_id, device):
    """Forward one view once; return class_fp32 and mask_quality records."""
    values = F.interpolate(pixels[None].to(device), size=(network_size, network_size),
                           mode="bilinear", align_corners=False)
    mask = F.interpolate(valid[None, None].to(device).float(),
                         size=(network_size, network_size), mode="nearest")[:, 0].bool()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(pixel_values=values, pixel_mask=mask)
    shared = decode_records(output, [image_id], SCORE_THRESHOLD, MASK_THRESHOLD,
                            output_size, score_mode="class_fp32", _include_quality=True)
    pure, quality = [], []
    for record in shared:
        quality_score = record.pop("_quality_score")
        pure.append(record)
        if quality_score >= SCORE_THRESHOLD:
            quality.append({**record, "score": quality_score})
    return {"class_fp32": pure, "mask_quality": quality}


def accumulate_recall(total, addition):
    for group, values in addition.items():
        for key, value in values.items():
            total[group][key] += value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if not 8 <= args.images <= 32:
        parser.error("This probe is capped at 32 images")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(2)
    checkpoint = project_path(args.checkpoint)
    model, config, device = load_model_m2f(checkpoint, "cuda")
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(0.40)
    dataset = RGBDInstanceDataset(project_path(config["data_root"]), "val", mode=config["mode"],
                                  jitter=None, crop_size=None)
    truth = dataset.build_coco()
    annotations = defaultdict(list)
    for ann in truth.dataset["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations[int(ann["image_id"])].append(ann)
    indices = select_indices(dataset, annotations, args.images, args.seed)
    image_ids = [dataset.ids[index] for index in indices]

    source_names = ["m2f_rgbd/model.py", "m2f_rgbd/data.py", "m2f_rgbd/postprocess.py",
                    "m2f_rgbd/tiling.py", "tools/probe_tile_zoom.py", "predict_m2f.py"]
    write_json(out / "manifest.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "source_sha256": {name: digest(project_path(name)) for name in source_names},
        "images": image_ids, "seed": args.seed,
        "global_network_input": 1536, "global_output_size": 1024,
        "tile_network_input": 1024, "layouts": LAYOUTS, "score_modes": list(MODES),
        "score_threshold": SCORE_THRESHOLD, "mask_threshold": MASK_THRESHOLD,
        "mask_nms_iou": NMS_IOU, "maxDets": [1, 10, 100],
        "precision": "bf16 forward, fp32 mask interpolation and instance scores"})

    configs = [(f"{name}_{mode}", name, mode) for name in ("global", *LAYOUTS) for mode in MODES]
    records = {name: [] for name, _, _ in configs}
    recall = {name: {"small": defaultdict(int), "tiny": defaultdict(int)} for name, _, _ in configs}
    started = time.monotonic()
    for done, index in enumerate(indices, 1):
        item = dataset[index]
        image_id = int(item["image_id"])
        global_variants = view_variants(model, item["pixel_values"], item["pixel_mask"],
                                        1536, 1024, image_id, device)
        per_image = {f"global_{mode}": box_nms(global_variants[mode]) for mode in MODES}
        for name, spec in LAYOUTS.items():
            positions = [(x, y) for y in tile_positions(1024, spec["tile"], spec["stride"])
                         for x in tile_positions(1024, spec["tile"], spec["stride"])]
            union = {mode: list(global_variants[mode]) for mode in MODES}
            for tile_index in spec["indices"]:
                x, y = positions[tile_index]
                local = view_variants(model,
                                      item["pixel_values"][:, y:y + spec["tile"], x:x + spec["tile"]],
                                      item["pixel_mask"][y:y + spec["tile"], x:x + spec["tile"]],
                                      1024, spec["tile"], image_id, device)
                for mode in MODES:
                    for record in local[mode]:
                        mapped = map_tile_record(record, x, y, spec["tile"])
                        if mapped is not None:
                            union[mode].append(mapped)
            for mode in MODES:
                per_image[f"{name}_{mode}"] = mask_nms(union[mode], NMS_IOU, max_dets=100)
        for key, values in per_image.items():
            records[key].extend(values)
            accumulate_recall(recall[key], candidate_recall(values, annotations[image_id]))
        if done == 1 or done % 4 == 0:
            print(f"{done}/{len(indices)} images elapsed={time.monotonic() - started:.1f}s", flush=True)
        del item, global_variants, per_image

    results = {}
    for name, _, mode in configs:
        metrics = evaluate_predictions(truth, records[name], image_ids)
        segm = metrics["segm"]
        small, tiny = recall[name]["small"], recall[name]["tiny"]
        results[name] = {
            "layout": name.rsplit("_", 1)[0], "mode": mode,
            "AP": segm["AP"], "AP50": segm["AP50"], "AP75": segm["AP75"],
            "AP_s": segm["AP_s"], "AP_m": segm["AP_m"], "AP_l": segm["AP_l"],
            "detections": len(records[name]),
            "small_recall_0.50": small["matched_0.50"] / max(small["gt"], 1),
            "tiny_recall_0.50": tiny["matched_0.50"] / max(tiny["gt"], 1),
            "tiny_recall_0.75": tiny["matched_0.75"] / max(tiny["gt"], 1),
            "tiny_gt": tiny["gt"],
        }
        print(f"{name:24s} AP={segm['AP']:.4f} AP_s={segm['AP_s']:.4f} "
              f"tiny_R50={results[name]['tiny_recall_0.50']:.3f}", flush=True)
        with gzip.open(out / f"{name}.json.gz", "wt") as stream:
            json.dump(records[name], stream)
    write_json(out / "results.json", {"images": len(indices), "seconds": time.monotonic() - started,
                                      "results": results})
    print("PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
