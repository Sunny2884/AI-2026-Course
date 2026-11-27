"""常规 PyTorch 训练循环、COCO 实例分割评价和模型测速。"""
from contextlib import nullcontext
from copy import deepcopy
import time

import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

METRIC_NAMES = ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
                "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l")


def train_one_epoch(model, loader, optimizer, scaler, device, amp=False):
    """用 loader 训练 model 一轮，返回按全部 rank 样本数平均的 loss 字典。

    参数 optimizer/scaler 负责优化与梯度缩放；device 是本进程设备，amp 在 CUDA
    上启用 FP16。model 可为普通模型或 DDP，各 rank 的 loader 必须具有相同批次数。
    """
    model.train()
    distributed = dist.is_available() and dist.is_initialized()
    main_process = not distributed or dist.get_rank() == 0
    amp = amp and device.type == "cuda"
    totals = {}
    images_seen = torch.zeros((), device=device)
    progress = tqdm(loader, desc="Train", disable=not main_process)
    for step, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        # 仅框、类别、掩码参与损失；image_id、area 等记录字段保留在 CPU。
        targets = [{key: value.to(device, non_blocking=True)
                    if key in {"boxes", "labels", "masks"} else value
                    for key, value in target.items()}
                   for target in targets]
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            losses = model(images, targets)
            loss = sum(losses.values())
        # 所有 rank 一起停止，避免单个 rank 抛错后其它 rank 卡在 DDP 反传。
        finite_loss = torch.isfinite(loss.detach()).to(torch.int32)
        if distributed:
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
        if not finite_loss.item():
            raise FloatingPointError("至少一个训练进程出现非有限 loss，全部进程停止本轮训练")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        finite_norm = torch.isfinite(gradient_norm).to(torch.int32)
        if distributed:
            dist.all_reduce(finite_norm, op=dist.ReduceOp.MIN)
        if finite_norm.item():
            scaler.step(optimizer)
            scaler.update()
        elif amp:
            # 有限梯度的范数求和也可能溢出；不能只依靠 unscale_ 的逐元素检查。
            # 全部 rank 跳过本次更新并同步降低 scale，下一批会重新清空梯度。
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
            # 手动降低 scale 后，连续成功步数也要重置，避免下一步立即再次升高。
            scaler_state = scaler.state_dict()
            scaler_state["_growth_tracker"] = 0
            scaler.load_state_dict(scaler_state)
            if main_process:
                progress.write("梯度范数溢出，已跳过本批参数更新并降低 AMP scale")
        else:
            raise FloatingPointError("至少一个训练进程出现非有限梯度范数")
        for name, value in losses.items():
            if name not in totals:
                totals[name] = torch.zeros((), device=device)
            totals[name].add_(value.detach().float(), alpha=len(images))
        images_seen += len(images)
        if main_process and step % 20 == 0:
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        del images, targets, losses, loss
    # 整轮仅合并一次各项 loss 总和与样本数，随后统一回传 CPU。
    statistics = torch.stack([*totals.values(), images_seen])
    if distributed:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    values = statistics.cpu().tolist()
    return {name: value / values[-1] for name, value in zip(totals, values[:-1])}


@torch.inference_mode()
def collect_predictions(model, loader, device, score_threshold=0.001, quiet=False, amp=False):
    """对 loader 推理并按 score_threshold 筛选，返回原图坐标的 COCO results 列表。

    参数 model 是普通模型或 DDP.module，device 为本进程设备；quiet 关闭进度条，
    amp 仅在 CUDA 上启用 FP16。函数不进行跨 rank 通信，空 loader 返回空列表。
    """
    model.eval()
    records = []
    for images, targets in tqdm(loader, desc="Predict", disable=quiet):
        context = torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
        with context:
            outputs = model([image.to(device, non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"])
            # 框、置信度和类别一次回传；先在 CPU 选出需要导出的实例。
            metadata = torch.cat([
                output["boxes"], output["scores"].unsqueeze(1), output["labels"].unsqueeze(1),
            ], dim=1).detach().cpu().tolist()
            selected = [index for index, item in enumerate(metadata)
                        if item[5] == 1 and item[4] >= score_threshold]
            if selected:
                # 每张图只回传一次 uint8 掩码，最多约 100 MiB；避免复制浮点掩码。
                binary_masks = (output["masks"][:, 0] >= 0.5).to(torch.uint8).cpu().numpy()
                for index in selected:
                    binary = binary_masks[index]
                    x1, y1, x2, y2, score, _ = metadata[index]
                    rle = mask_utils.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("ascii")
                    records.append({"image_id": image_id, "category_id": 1, "score": score,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": rle})
                del binary_masks, binary
            # 循环变量也会持有 GPU 张量；下一批 forward 前一并释放。
            del output
        del outputs
    return records


def evaluate_predictions(ground_truth, records, image_ids):
    """对 COCO ground_truth、预测 records 和 image_ids 计算并返回 mask/bbox 指标字典。"""
    metrics = {}
    for iou_type in ("segm", "bbox"):
        # loadRes 遇到 bbox 会使用框面积；mask AP_s 必须按 mask 面积划分预测。
        geometry = "segmentation" if iou_type == "segm" else "bbox"
        results = [{key: record[key] for key in ("image_id", "category_id", "score", geometry)}
                   for record in records]
        if results:
            detections = ground_truth.loadRes(results)
        else:
            # COCO.loadRes([]) 会报错，因此显式构造没有预测的结果对象。
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


@torch.inference_mode()
def benchmark_model(model, dataset, device, images=20):
    """在 dataset 上以 batch=1 测 model；返回参数量、FP32 大小、平均延迟和 FPS。"""
    model.eval()
    first = dataset[0][0].to(device)
    for _ in range(3):
        model([first])
    latencies = []
    for index in np.linspace(0, len(dataset) - 1, min(images, len(dataset)), dtype=int):
        image = dataset[int(index)][0].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model([image])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        del output
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": parameters, "parameter_mib_fp32": parameters * 4 / 1024**2,
            "mean_latency_ms": float(np.mean(latencies) * 1000),
            "fps": float(1 / np.mean(latencies)), "images": len(latencies),
            "device": str(device), "precision": "float32", "batch_size": 1,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "timing_scope": "model forward, including resize/NMS/mask pasting; excludes disk I/O, transfers and RLE"}
