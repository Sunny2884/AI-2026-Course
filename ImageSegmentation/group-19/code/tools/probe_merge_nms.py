"""CPU-only merge ablation over the frozen tiles5 validation cache.

The cached per-image records are already merged at mask IoU 0.50; this probe asks
whether a second suppression pass (box NMS, or stricter mask NMS) removes
duplicate detections that survived the first merge. No GPU, no new forwards.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import torch
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.tiling import mask_nms
from m2f_rgbd.utils import project_path, write_json

CONFIGS = {
    "cached_mask0.50": ("none", 0.0),
    "box_0.70": ("box", 0.70),
    "box_0.65": ("box", 0.65),
    "box_0.60": ("box", 0.60),
    "box_0.55": ("box", 0.55),
    "box_0.50": ("box", 0.50),
    "mask_0.45": ("mask", 0.45),
}


def box_suppress(records, iou_threshold, max_dets=100):
    if not records:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in
                          [record["bbox"] for record in records]], dtype=torch.float32)
    scores = torch.tensor([record["score"] for record in records], dtype=torch.float32)
    keep = nms(boxes, scores, iou_threshold)[:max_dets]
    return [records[index] for index in keep.tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--phase", default="val")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    signature = manifest["signature"]
    phase = root / args.phase
    image_ids = json.loads((phase / "image_ids.json").read_text())
    print(f"{len(image_ids)} images in {phase}", flush=True)

    dataset = RGBDInstanceDataset(project_path("dataset"), "val", jitter=None, crop_size=None)
    truth = dataset.build_coco()

    cached = {}
    for count, image_id in enumerate(image_ids, 1):
        with gzip.open(phase / "items" / f"{image_id}.json.gz", "rt") as stream:
            item = json.load(stream)
        if item["image_id"] != image_id or item["signature"] != signature:
            raise ValueError(f"Cache provenance mismatch: {image_id}")
        cached[image_id] = item["tiles5"]
        if count % 500 == 0:
            print(f"loaded {count}/{len(image_ids)}", flush=True)

    results = {}
    for name, (kind, threshold) in CONFIGS.items():
        records = []
        for image_id in image_ids:
            branch = cached[image_id]
            if kind == "box":
                branch = box_suppress(branch, threshold)
            elif kind == "mask":
                branch = mask_nms(branch, threshold, max_dets=100)
            records.extend(branch)
        metrics = evaluate_predictions(truth, records, image_ids)
        segm = metrics["segm"]
        results[name] = {"kind": kind, "threshold": threshold, "instances": len(records),
                         "segm": segm, "bbox_AP": metrics["bbox"]["AP"]}
        print(f"{name:16s} AP={segm['AP']:.4f} AP50={segm['AP50']:.4f} "
              f"AP75={segm['AP75']:.4f} AP_s={segm['AP_s']:.4f} "
              f"AP_m={segm['AP_m']:.4f} dets={len(records)}", flush=True)
        del records
    output = Path(args.output).resolve()
    write_json(output / f"merge_{args.phase}.json", {"images": len(image_ids), "results": results})
    print("MERGE_ABLATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
