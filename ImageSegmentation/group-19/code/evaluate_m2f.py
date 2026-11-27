"""用训练好的 Mask2Former 在 val 上评价并统计模型大小与推理速度。

指标协议与旧管线 evaluate.py 完全一致；--benchmark 输出参数量、FP32 体积、
平均延迟和 FPS，供报告使用。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2f_rgbd.data import RGBDInstanceDataset, collate_targets
from m2f_rgbd.engine import collect_predictions, collect_predictions_ms_tta, evaluate_predictions
from m2f_rgbd.utils import load_checkpoint, make_loader, project_path, setup_device, write_json
from predict_m2f import load_model_m2f


@torch.inference_mode()
def benchmark_model(model, dataset, device, images=20):
    """在 dataset 上以 batch=1 测 model；返回参数量、FP32 大小、平均延迟和 FPS。"""
    model.eval()
    first = dataset[0]["pixel_values"].to(device)
    first_mask = dataset[0]["pixel_mask"].to(device)
    for _ in range(3):
        model(pixel_values=first[None], pixel_mask=first_mask[None])
    latencies = []
    for index in np.linspace(0, len(dataset) - 1, min(images, len(dataset)), dtype=int):
        item = dataset[int(index)]
        image = item["pixel_values"].to(device)[None]
        mask = item["pixel_mask"].to(device)[None]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model(pixel_values=image, pixel_mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        del output
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": parameters, "parameter_mib_fp32": parameters * 4 / 1024**2,
            "mean_latency_ms": float(np.mean(latencies) * 1000),
            "fps": float(1 / np.mean(latencies)), "images": len(latencies),
            "device": str(device), "precision": "bfloat16 forward",
            "batch_size": 1,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "timing_scope": "model forward incl. mask/class decoding; excludes disk I/O, transfers and RLE"}


def main():
    """对 val 全量评价，可选统计模型大小与速度；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="N>0 只用前 N 张 val 图（快速诊断）")
    parser.add_argument("--stride", type=int, default=1, help=">1 时等间隔抽样 val 图像（诊断用）")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-images", type=int, default=20)
    parser.add_argument("--score-mode", choices=["legacy_bf16", "class_fp32", "mask_quality"],
                        default="class_fp32", help="Instance ranking; use mask_quality for Meta's mask-aware score")
    parser.add_argument("--tta", action="store_true", help="多尺度 + 水平翻转 TTA")
    parser.add_argument("--tta-scales", default="1.0,1.25", help="--tta 使用的尺度列表")
    parser.add_argument("--no-flip", action="store_true", help="--tta 时只做多尺度，不做水平翻转")
    parser.add_argument("--output", default="outputs/evaluation_m2f")
    args = parser.parse_args()
    if args.tta and args.score_mode != "class_fp32":
        parser.error("Selected score mode is currently validated for single-pass inference only; omit --tta")
    model, config, device = load_model_m2f(args.checkpoint, args.device)
    root = project_path(args.data_root or config["data_root"])
    full_dataset = RGBDInstanceDataset(root, "val", mode=config["mode"],
                                       jitter=None, crop_size=None, limit=args.limit)
    if args.stride > 1:
        indices = list(range(0, len(full_dataset), args.stride))
        image_ids = [full_dataset.ids[index] for index in indices]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset
        image_ids = full_dataset.ids
    loader = make_loader(dataset, args.batch_size, args.workers, collate_fn=collate_targets)
    if args.tta:
        scales = tuple(float(scale) for scale in args.tta_scales.split(",") if scale.strip())
        records = collect_predictions_ms_tta(model, loader, device, config["score_threshold"],
                                             amp_dtype=config["amp_dtype"], scales=scales,
                                             flip=not args.no_flip)
    else:
        records = collect_predictions(model, loader, device, config["score_threshold"],
                                      amp_dtype=config["amp_dtype"], score_mode=args.score_mode)
    ground_truth = full_dataset.build_coco()
    metrics = evaluate_predictions(ground_truth, records, image_ids)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    summary = {"checkpoint": str(project_path(args.checkpoint)),
               "images": len(dataset), "instances": len(records), "metrics": metrics,
               "score_mode": args.score_mode,
               "tta": (f"scales={args.tta_scales}," + ("no-flip" if args.no_flip else "flip"))
               if args.tta else None}
    if args.benchmark:
        benchmark = benchmark_model(model, dataset, device, args.benchmark_images)
        summary["benchmark"] = benchmark
        write_json(output / "benchmark.json", benchmark)
    write_json(output / "summary.json", summary)
    print(f"Mask AP={metrics['segm']['AP']:.4f}; AP_s={metrics['segm']['AP_s']:.4f}; "
          f"AP50={metrics['segm']['AP50']:.4f}; AP75={metrics['segm']['AP75']:.4f}")
    if args.benchmark:
        print(f"params={benchmark['parameters']}; fp32={benchmark['parameter_mib_fp32']:.1f} MiB; "
              f"latency={benchmark['mean_latency_ms']:.1f} ms; fps={benchmark['fps']:.2f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
