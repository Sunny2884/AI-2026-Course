"""Single-class Mask2Former instance decoding with explicit scoring provenance."""
import math

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.nn import functional as F

SCORE_MODES = ("legacy_bf16", "class_fp32", "mask_quality")


def decode_records(outputs, image_ids, score_threshold=0.001, mask_threshold=0.5,
                   output_size=1024, score_mode="mask_quality", chunk_size=16,
                   _include_quality=False):
    """Decode independent query masks; never flatten them to a panoptic map.

    legacy_bf16 reproduces the old bf16 softmax/interpolation for comparison.
    class_fp32 uses float32 classification and mask postprocessing.
    mask_quality multiplies the float32 class score by mean foreground mask
    probability, following Meta's instance inference. All thresholds and output
    geometry are shared between evaluation and CSV export.
    """
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unknown score_mode: {score_mode}")
    if not 0 <= score_threshold <= 1 or not 0 < mask_threshold < 1:
        raise ValueError("Invalid score or mask threshold")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
    classes, masks = outputs.class_queries_logits, outputs.masks_queries_logits
    if classes.ndim != 3 or classes.shape[-1] != 2:
        raise ValueError("Expected class logits [B,Q,2] for foreground/no-object")
    if masks.ndim != 4 or tuple(masks.shape[:2]) != tuple(classes.shape[:2]):
        raise ValueError("Mask and class query shapes disagree")
    if len(image_ids) != classes.shape[0]:
        raise ValueError("Image IDs do not match batch size")
    if not torch.isfinite(classes).all() or not torch.isfinite(masks).all():
        raise FloatingPointError("Non-finite model logits")
    if score_mode == "legacy_bf16":
        class_scores = classes.to(torch.bfloat16).softmax(-1)[..., 0].float()
    else:
        logits = classes.float()
        class_scores = (logits[..., 0] - logits[..., 1]).sigmoid()
    records = []
    for batch_index, image_id in enumerate(image_ids):
        selected = torch.where(class_scores[batch_index] >= score_threshold)[0]
        for start in range(0, len(selected), chunk_size):
            query_ids = selected[start:start + chunk_size]
            logits = masks[batch_index, query_ids]
            logits = logits.to(torch.bfloat16) if score_mode == "legacy_bf16" else logits.float()
            if tuple(logits.shape[-2:]) != size:
                logits = F.interpolate(logits[None], size=size, mode="bilinear",
                                       align_corners=False)[0]
            probabilities = logits.sigmoid()
            binary = probabilities > mask_threshold
            scores = class_scores[batch_index, query_ids]
            quality_scores = None
            if score_mode == "mask_quality" or _include_quality:
                area = binary.flatten(1).sum(1)
                confidence_sum = (probabilities * binary).flatten(1).sum(1)
                quality = confidence_sum / (area.float() + 1e-6)
                quality_scores = scores * quality
                if score_mode == "mask_quality":
                    scores = quality_scores
            binary_np = binary.to(torch.uint8).cpu().numpy()
            scores_cpu = scores.float().cpu().tolist()
            quality_cpu = quality_scores.float().cpu().tolist() if _include_quality else [None] * len(scores_cpu)
            for mask, score, quality_score in zip(binary_np, scores_cpu, quality_cpu):
                if score < score_threshold or not mask.any():
                    continue
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite instance score")
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")
                record = {"image_id": int(image_id), "category_id": 1,
                                "score": float(score),
                                "bbox": mask_utils.toBbox(rle).astype(float).tolist(),
                                "segmentation": rle}
                if _include_quality:
                    record["_quality_score"] = float(quality_score)
                records.append(record)
    return records


def decode_record_variants(outputs, image_ids, score_threshold=0.001,
                           mask_threshold=0.5, output_size=1024):
    """Reuse FP32 mask geometry and RLE encoding for the two FP32 scorers."""
    legacy = decode_records(outputs, image_ids, score_threshold, mask_threshold,
                            output_size, score_mode="legacy_bf16")
    shared = decode_records(outputs, image_ids, score_threshold, mask_threshold,
                            output_size, score_mode="class_fp32", _include_quality=True)
    pure, quality = [], []
    for item in shared:
        quality_score = item.pop("_quality_score")
        pure.append(item)
        if quality_score >= score_threshold:
            quality.append({**item, "score": quality_score})
    return {"legacy_bf16": legacy, "class_fp32": pure, "mask_quality": quality}
