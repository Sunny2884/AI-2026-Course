"""Geometry-only utilities for an isolated high-resolution tiling probe."""
import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment


def tile_positions(length, tile_size, stride):
    if not 0 < stride <= tile_size <= length:
        raise ValueError("Require 0 < stride <= tile_size <= image length")
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def map_tile_record(record, x0, y0, tile_size, full_size=1024, edge_margin=1):
    """Drop interior-edge truncations; preserve masks touching image boundaries."""
    x, y, w, h = record["bbox"]
    if ((x0 > 0 and x <= edge_margin) or
        (y0 > 0 and y <= edge_margin) or
        (x0 + tile_size < full_size and x + w >= tile_size - edge_margin) or
        (y0 + tile_size < full_size and y + h >= tile_size - edge_margin)):
        return None
    patch = mu.decode(record["segmentation"])
    if patch.shape != (tile_size, tile_size):
        raise ValueError("Tile mask size mismatch")
    canvas = np.zeros((full_size, full_size), dtype=np.uint8, order="F")
    canvas[y0:y0 + tile_size, x0:x0 + tile_size] = patch
    rle = mu.encode(canvas)
    rle["counts"] = rle["counts"].decode("ascii")
    return {"image_id": record["image_id"], "category_id": 1,
            "score": record["score"], "bbox": [x + x0, y + y0, w, h],
            "segmentation": rle}


def mask_nms(records, iou_threshold=0.5, max_dets=100):
    """Suppress overlapping instance masks, preserving disjoint interleaved objects."""
    if not records:
        return []
    order = np.argsort(-np.array([r["score"] for r in records]), kind="stable")
    kept = []
    while len(order) and len(kept) < max_dets:
        chosen = int(order[0])
        kept.append(records[chosen])
        remaining = order[1:]
        if not len(remaining):
            break
        overlap = mu.iou([records[chosen]["segmentation"]],
                         [records[int(i)]["segmentation"] for i in remaining],
                         [0] * len(remaining))[0]
        order = remaining[overlap <= iou_threshold]
    return kept


def max_matching_count(ious, threshold):
    if ious.size == 0:
        return 0
    valid = (ious >= threshold).astype(np.int32)
    rows, cols = linear_sum_assignment(-valid)
    return int(valid[rows, cols].sum())


def candidate_recall(records, annotations, thresholds=(0.5, 0.75, 0.9)):
    """Score-free, one-to-one mask recall: a diagnostic, not AP."""
    groups = {"small": [a for a in annotations if 0 < a["area"] < 1024],
              "tiny": [a for a in annotations if 0 < a["area"] < 256]}
    result = {}
    for name, selected in groups.items():
        gt = []
        for ann in selected:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                seg = mu.merge(mu.frPyObjects(seg, 1024, 1024))
            elif isinstance(seg["counts"], list):
                seg = mu.frPyObjects(seg, 1024, 1024)
            gt.append(seg)
        ious = (mu.iou([r["segmentation"] for r in records], gt, [0] * len(gt))
                if records and gt else np.zeros((len(records), len(gt))))
        result[name] = {"gt": len(gt), **{f"matched_{t:.2f}": max_matching_count(ious, t)
                                          for t in thresholds}}
    return result
