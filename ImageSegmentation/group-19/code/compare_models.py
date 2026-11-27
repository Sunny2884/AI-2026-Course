"""基线与改进模型统一协议对比：val 指标、模型大小、推理速度。

两个模型用完全相同的 COCOeval 协议在完整 val 上评价；速度均为 batch=1
（基线 FP32，改进模型 bf16 前向）。输出合并表格到 --output 目录，供报告使用。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

METRIC_KEYS = ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l")


def run_baseline(checkpoint, root, device):
    """旧管线：全 val 评价 + 测速；返回 (metrics, benchmark)。"""
    import torch
    from rgbdseg.data import ComponentDataset
    from rgbdseg.engine import benchmark_model, collect_predictions, evaluate_predictions
    from rgbdseg.utils import load_model, make_loader

    model, config, device_obj = load_model(checkpoint, device)
    dataset = ComponentDataset(root, "val", use_depth=config["use_depth"], limit=0)
    loader = make_loader(dataset, 1, 2)
    records = collect_predictions(model, loader, device_obj, config.get("score_threshold", 0.001), amp=False)
    metrics = evaluate_predictions(dataset.build_coco(), records, dataset.ids)
    benchmark = benchmark_model(model, dataset, device_obj, images=20)
    del model
    torch.cuda.empty_cache()
    return metrics, benchmark


def run_improved(checkpoint, root, device):
    """新管线：全 val 评价 + 测速；返回 (metrics, benchmark)。"""
    import torch
    from m2f_rgbd.data import RGBDInstanceDataset, collate_targets
    from m2f_rgbd.engine import collect_predictions, evaluate_predictions
    from m2f_rgbd.utils import make_loader
    from predict_m2f import load_model_m2f
    from evaluate_m2f import benchmark_model

    model, config, device_obj = load_model_m2f(checkpoint, device)
    dataset = RGBDInstanceDataset(root, "val", mode=config["mode"], jitter=None, crop_size=None)
    loader = make_loader(dataset, 2, 4, collate_fn=collate_targets)
    records = collect_predictions(model, loader, device_obj, config["score_threshold"],
                                  amp_dtype=config["amp_dtype"])
    metrics = evaluate_predictions(dataset.build_coco(), records, dataset.ids)
    benchmark = benchmark_model(model, dataset, device_obj, images=20)
    del model
    torch.cuda.empty_cache()
    return metrics, benchmark


def main():
    """运行两组评价并输出合并表格；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", default="outputs/baseline/best.pth")
    parser.add_argument("--improved-checkpoint", required=True)
    parser.add_argument("--data-root", default="./dataset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="outputs/comparison")
    args = parser.parse_args()
    root = Path(args.data_root)

    base_metrics, base_bench = run_baseline(args.baseline_checkpoint, root, args.device)
    improved_metrics, improved_bench = run_improved(args.improved_checkpoint, root, args.device)

    table = {}
    for name in METRIC_KEYS:
        table[f"segm_{name}"] = {
            "baseline": base_metrics["segm"][name],
            "improved": improved_metrics["segm"][name],
            "delta": improved_metrics["segm"][name] - base_metrics["segm"][name],
        }
    table["parameters"] = {"baseline": base_bench["parameters"], "improved": improved_bench["parameters"]}
    table["parameter_mib_fp32"] = {"baseline": base_bench["parameter_mib_fp32"],
                                   "improved": improved_bench["parameter_mib_fp32"]}
    table["latency_ms"] = {"baseline": base_bench["mean_latency_ms"],
                           "improved": improved_bench["mean_latency_ms"]}
    table["fps"] = {"baseline": base_bench["fps"], "improved": improved_bench["fps"]}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    print(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"Saved {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
