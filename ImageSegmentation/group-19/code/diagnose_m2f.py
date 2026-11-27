"""Mask2Former 小目标错误分析：漏检/框对掩码错/边界精度与逐 IoU 的 AP_s。

在 val 等间隔子集上运行，输出：
- AP_s 在 IoU 0.5/0.6/0.7/0.75/0.8/0.9 上的值（判断是“找不到”还是“掩码糊”）；
- 小目标 GT 的匹配情况：IoU>=0.5 匹配、框对但掩码 IoU<0.5、完全漏检；
- 匹配小目标的掩码 IoU 中位数与 IoU>=0.75 占比；
- 预测/GT 面积比（过分割或欠分割）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2f_rgbd.data import RGBDInstanceDataset, collate_targets, decode_mask
from m2f_rgbd.engine import collect_predictions, evaluate_predictions
from m2f_rgbd.utils import make_loader, project_path, setup_device, write_json
from predict_m2f import load_model_m2f

SMALL_AREA = 1024
BUCKETS = ((0, 256), (256, 512), (512, 1024))


def mask_iou(mask_a, mask_b):
    """两个二值掩码的 IoU；均为空时返回 0。"""
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def box_iou(box_a, box_b):
    """两个 [x1,y1,x2,y2] 框的 IoU；均无面积时返回 0。"""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def ap_small_by_iou(ground_truth, records, image_ids):
    """返回 {IoU 阈值: AP_small}，复用 COCOeval 的精度张量。"""
    from pycocotools.cocoeval import COCOeval
    results = [{"image_id": record["image_id"], "category_id": 1,
                "score": record["score"], "segmentation": record["segmentation"]}
               for record in records]
    detections = ground_truth.loadRes(results)
    evaluator = COCOeval(ground_truth, detections, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # [T, R, K, A, M]
    small_index = list(evaluator.params.areaRngLbl).index("small")
    output = {}
    for threshold_index, threshold in enumerate(evaluator.params.iouThrs):
        values = precision[threshold_index, :, :, small_index, :]
        valid = values[values > -1]
        output[f"{threshold:.2f}"] = float(valid.mean()) if valid.size else -1.0
    return output


def main():
    """运行预测、匹配统计与逐 IoU AP_s，打印并保存 JSON；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="outputs/diagnostics_m2f")
    args = parser.parse_args()

    model, config, device = load_model_m2f(args.checkpoint, args.device)
    root = project_path(args.data_root or config["data_root"])
    dataset = RGBDInstanceDataset(root, "val", mode=config["mode"], jitter=None, crop_size=None)
    indices = list(range(0, len(dataset), max(1, args.stride)))
    subset = torch.utils.data.Subset(dataset, indices)
    image_ids = [dataset.ids[index] for index in indices]
    loader = make_loader(subset, args.batch_size, args.workers, collate_fn=collate_targets)
    records = collect_predictions(model, loader, device, config["score_threshold"],
                                  amp_dtype=config["amp_dtype"])
    ground_truth = dataset.build_coco()
    by_image = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    from pycocotools import mask as mask_utils

    stats = {"small_total": 0, "small_matched": 0, "small_box_only": 0, "small_missed": 0,
             "buckets": {f"{lo}-{hi}": {"total": 0, "matched": 0} for lo, hi in BUCKETS}}
    ious, area_ratios = [], []
    annotation_index = {}
    for annotation in ground_truth.dataset["annotations"]:
        annotation_index.setdefault(int(annotation["image_id"]), []).append(annotation)
    for image_id in image_ids:
        annotations = [item for item in annotation_index.get(int(image_id), [])
                       if item["area"] < SMALL_AREA]
        if not annotations:
            continue
        predictions = sorted(by_image.get(int(image_id), []),
                             key=lambda record: record["score"], reverse=True)[:100]
        pred_masks = [mask_utils.decode(record["segmentation"]).astype(bool)
                      for record in predictions]
        pred_boxes = [record["bbox"] for record in predictions]
        for annotation in annotations:
            gt_mask = decode_mask(annotation["segmentation"], 1024, 1024).astype(bool)
            gt_box = (annotation["bbox"][0], annotation["bbox"][1],
                      annotation["bbox"][0] + annotation["bbox"][2],
                      annotation["bbox"][1] + annotation["bbox"][3])
            stats["small_total"] += 1
            for lo, hi in BUCKETS:
                if lo <= annotation["area"] < hi:
                    stats["buckets"][f"{lo}-{hi}"]["total"] += 1
            best_iou, best_index = 0.0, -1
            best_box_iou, best_box_index = 0.0, -1
            for index, (pred_mask, pred_box) in enumerate(zip(pred_masks, pred_boxes)):
                iou = mask_iou(gt_mask, pred_mask)
                if iou > best_iou:
                    best_iou, best_index = iou, index
                overlap = box_iou(gt_box, (pred_box[0], pred_box[1],
                                           pred_box[0] + pred_box[2],
                                           pred_box[1] + pred_box[3]))
                if overlap > best_box_iou:
                    best_box_iou, best_box_index = overlap, index
            if best_iou >= 0.5:
                stats["small_matched"] += 1
                for lo, hi in BUCKETS:
                    if lo <= annotation["area"] < hi:
                        stats["buckets"][f"{lo}-{hi}"]["matched"] += 1
                ious.append(best_iou)
                pred_area = pred_masks[best_index].sum()
                area_ratios.append(float(pred_area / max(annotation["area"], 1)))
            elif best_box_iou >= 0.5:
                stats["small_box_only"] += 1
            else:
                stats["small_missed"] += 1
    summary = {
        "images": len(image_ids),
        "small_iou_matched": float(np.mean(ious)) if ious else 0.0,
        "small_iou_median": float(np.median(ious)) if ious else 0.0,
        "small_iou_ge_075": float(np.mean(np.array(ious) >= 0.75)) if ious else 0.0,
        "area_ratio_mean": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "area_ratio_median": float(np.median(area_ratios)) if area_ratios else 0.0,
        "ap_small_by_iou": ap_small_by_iou(ground_truth, records, image_ids),
        "detections": len(records),
    }
    summary.update(stats)
    output = project_path(args.output)
    write_json(output / "diagnostics.json", summary)
    print(json_dumps(summary))


def json_dumps(value):
    """缩进 JSON 字符串，便于日志查看。"""
    import json
    return json.dumps(value, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
