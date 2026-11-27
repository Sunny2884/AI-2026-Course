"""Mask2Former 训练循环、推理后处理与 COCO 评价。

评价协议与旧管线 rgbdseg.engine.evaluate_predictions 完全一致（同 COCOeval
参数、同 0~1 指标口径），保证基线与新模型可直接比较。
"""
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

METRIC_NAMES = ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
                "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l")


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    amp_dtype="bf16", grad_accum=1, max_grad_norm=0.01,
                    warmup_iters=0, total_iters=0):
    """训练一轮；返回按全部 rank 样本数平均的 loss 字典。

    使用 autocast(amp_dtype) 混合精度（bf16 无需 GradScaler）；每 grad_accum
    个 micro-batch 执行一次 unscale/clip/step；学习率按成功更新步数在
    warmup_iters 内线性升温，之后由外部 scheduler 控制（见 train_m2f.py）。
    """
    model.train()
    distributed = dist.is_available() and dist.is_initialized()
    main_process = not distributed or dist.get_rank() == 0
    dtype = {"bf16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype)
    totals = {}
    images_seen = torch.zeros((), device=device)
    progress = tqdm(loader, desc=f"Train e{epoch}", disable=not main_process)
    steps = 0
    for batch in progress:
        pixel_values, pixel_mask, class_labels, mask_labels, _ = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        class_labels = [labels.to(device, non_blocking=True) for labels in class_labels]
        mask_labels = [masks.to(device, non_blocking=True) for masks in mask_labels]

        context = torch.autocast("cuda", dtype=dtype) if dtype is not None and device.type == "cuda" else nullcontext()
        with context:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                            class_labels=class_labels, mask_labels=mask_labels)
            loss = outputs.loss
        finite_loss = torch.isfinite(loss.detach()).to(torch.int32)
        if distributed:
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
        if not finite_loss.item():
            raise FloatingPointError("至少一个训练进程出现非有限 loss，全部进程停止本轮训练")

        (loss / grad_accum).backward()
        steps += 1
        if steps % grad_accum == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        for name, value in outputs.items():
            if name.startswith("loss") and value is not None and torch.is_tensor(value):
                key = "loss" if name == "loss" else name
                if key not in totals:
                    totals[key] = torch.zeros((), device=device)
                totals[key].add_(value.detach().float(), alpha=len(batch))
        images_seen += len(batch)
        if main_process and steps % 10 == 0:
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        del pixel_values, pixel_mask, class_labels, mask_labels, outputs, loss
    statistics = torch.stack([*totals.values(), images_seen])
    if distributed:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    values = statistics.cpu().tolist()
    return {name: value / values[-1] for name, value in zip(totals, values[:-1])}


@torch.inference_mode()
def collect_predictions(model, loader, device, score_threshold=0.001, quiet=False,
                        amp_dtype="bf16", mask_threshold=0.5, output_size=1024,
                        score_mode="class_fp32"):
    """对 loader 推理并返回原图坐标的 COCO results 列表。

    后处理：class logits 做 softmax 后取前景类概率作为分数；mask logits 双线性
    上采样到原图分辨率后按 0.5 二值化；框由掩码外接矩形得到。
    """
    model.eval()
    records = []
    for batch in tqdm(loader, desc="Predict", disable=quiet):
        pixel_values, pixel_mask, _, _, image_ids = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        dtype = {"bf16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype)
        context = torch.autocast("cuda", dtype=dtype) if dtype is not None and device.type == "cuda" else nullcontext()
        with context:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        records.extend(outputs_to_records(outputs, image_ids, score_threshold,
                                          mask_threshold, output_size, score_mode=score_mode))
        del pixel_values, pixel_mask, outputs
    return records


def foreground_scores(class_queries_logits):
    """把类别 logits 转成前景分数，保留排序细节（避免 bf16 softmax 饱和）。

    单类 Mask2Former 的 class head 输出 [B,Q,2]（前景 / 无目标）。直接对
    bf16 logits 做 softmax 时，高置信查询的概率会舍入成精确的 1.0，实测提交
    文件里 35.9% 的检测分数并列，AP 依赖的排序信息被抹掉。这里先在 float32
    中计算 log-odds，再用 float32 sigmoid 映射回 [0,1)，数值仍可直接写入
    提交 CSV，但同一个 1.0 桶里的检测能被区分开。
    """
    logits = class_queries_logits.float()
    if logits.shape[-1] != 2:
        raise ValueError("foreground_scores 目前只支持单类（2 个 logits）输出")
    return (logits[..., 0] - logits[..., 1]).sigmoid()


def _record_from_mask(image_id, mask, score):
    """把单张二值掩码转换为一条 COCO results 记录。"""
    ys, xs = np.nonzero(mask)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"image_id": image_id, "category_id": 1, "score": float(score),
            "bbox": [float(xs.min()), float(ys.min()),
                     float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
            "segmentation": rle}


def _box_nms_merge(detections, iou_threshold=0.7, max_dets=100):
    """把多趟预测合并后用框 NMS 去重；detections 为 [(score, uint8 mask)]。

    Mask2Former 的查询槽位在不同尺度/翻转之间没有语义对应关系，不能按查询
    下标平均 logits，只能在还原到原图坐标后合并检测再做 NMS。
    """
    if not detections:
        return []
    from torchvision.ops import nms

    boxes, scores = [], []
    for score, mask in detections:
        ys, xs = np.nonzero(mask)
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)])
        scores.append(float(score))
    keep = nms(torch.tensor(boxes, dtype=torch.float32),
               torch.tensor(scores, dtype=torch.float32), iou_threshold)[:max_dets]
    return [detections[index] for index in keep.tolist()]


@torch.inference_mode()
def collect_predictions_tta(model, loader, device, score_threshold=0.001, quiet=False,
                            amp_dtype="bf16", mask_threshold=0.5, output_size=1024):
    """水平翻转 TTA：等价于单尺度合并式 TTA（见 collect_predictions_ms_tta）。"""
    return collect_predictions_ms_tta(model, loader, device, score_threshold, quiet,
                                      amp_dtype, mask_threshold, output_size,
                                      scales=(1.0,), flip=True)


@torch.inference_mode()
def collect_predictions_ms_tta(model, loader, device, score_threshold=0.001, quiet=False,
                               amp_dtype="bf16", mask_threshold=0.5, output_size=1024,
                               scales=(1.0, 1.25), flip=True, nms_iou=0.7):
    """多尺度 + 水平翻转 TTA：每趟独立后处理，在原图坐标合并后做框 NMS。

    各趟先各自解码为掩码（翻转趟把掩码翻转回原方向），再按分数做框 NMS
    合并。这样不同趟之间无需查询槽位对应，也不受查询排列随机性影响。
    """
    model.eval()
    records = []
    for batch in tqdm(loader, desc="Predict MS-TTA", disable=quiet):
        pixel_values, pixel_mask, _, _, image_ids = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        dtype = {"bf16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype)
        context = torch.autocast("cuda", dtype=dtype) if dtype is not None and device.type == "cuda" else nullcontext()
        detected = {int(image_id): [] for image_id in image_ids}
        for scale in scales:
            scaled_values, scaled_mask = pixel_values, pixel_mask
            if scale != 1.0:
                size = max(int(round(output_size * scale)), 32)
                scaled_values = torch.nn.functional.interpolate(
                    pixel_values, size=(size, size), mode="bilinear", align_corners=False)
                scaled_mask = torch.nn.functional.interpolate(
                    pixel_mask[:, None].float(), size=(size, size), mode="nearest")[:, 0].bool()
            for do_flip in ((True, False) if flip else (False,)):
                values = torch.flip(scaled_values, dims=[-1]) if do_flip else scaled_values
                mask = torch.flip(scaled_mask, dims=[-1]) if do_flip else scaled_mask
                with context:
                    outputs = model(pixel_values=values, pixel_mask=mask)
                scores_all = foreground_scores(outputs.class_queries_logits)
                msk = outputs.masks_queries_logits.float()
                if msk.shape[-2:] != (output_size, output_size):
                    msk = torch.nn.functional.interpolate(
                        msk, size=(output_size, output_size),
                        mode="bilinear", align_corners=False)
                if do_flip:
                    msk = torch.flip(msk, dims=[-1])
                binary = msk.sigmoid() > mask_threshold
                for index, image_id in enumerate(image_ids):
                    keep = torch.nonzero(scores_all[index] >= score_threshold).flatten()
                    masks = binary[index][keep].cpu().numpy()
                    scores = scores_all[index][keep].cpu().tolist()
                    for one_mask, score in zip(masks, scores):
                        if one_mask.any():
                            detected[int(image_id)].append((score, one_mask))
                del outputs, scores_all, msk, binary
        for image_id, detections in detected.items():
            for score, one_mask in _box_nms_merge(detections, nms_iou, max_dets=100):
                records.append(_record_from_mask(image_id, one_mask, score))
        del pixel_values, pixel_mask
    return records


def outputs_to_records(outputs, image_ids, score_threshold, mask_threshold, output_size,
                       score_mode="class_fp32"):
    """Shared FP32 instance decoder used by validation and submission export."""
    from .postprocess import decode_records
    return decode_records(outputs, image_ids, score_threshold, mask_threshold,
                          output_size, score_mode=score_mode)


def evaluate_predictions(ground_truth, records, image_ids):
    """对 COCO ground_truth、预测 records 和 image_ids 计算 mask/bbox 指标。

    与 rgbdseg.engine.evaluate_predictions 保持同一协议：segm 用分割几何、
    bbox 用框几何，maxDets=[1,10,100]，指标为 0~1。
    """
    metrics = {}
    for iou_type in ("segm", "bbox"):
        geometry = "segmentation" if iou_type == "segm" else "bbox"
        results = [{key: record[key] for key in ("image_id", "category_id", "score", geometry)}
                   for record in records]
        if results:
            detections = ground_truth.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {"images": deepcopy(ground_truth.dataset["images"]),
                                  "categories": [{"id": 1, "name": "component"}], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, iou_type)
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[iou_type] = dict(zip(METRIC_NAMES, map(float, evaluator.stats)))
    return metrics
