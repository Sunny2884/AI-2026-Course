"""在验证集计算 COCO Mask/Box AP，保存预测，可同时测量模型大小和速度。"""
import argparse

from rgbdseg.data import ComponentDataset
from rgbdseg.engine import benchmark_model, collect_predictions, evaluate_predictions
from rgbdseg.utils import load_model, make_loader, project_path, write_json


def main():
    """读取 checkpoint 和验证数据，保存指标/原图预测及可选测速结果；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--backbone-repo", help="Override the local DINOv3 source directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means the entire validation split")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-images", type=int, default=20)
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()
    if args.benchmark and args.benchmark_images < 1:
        parser.error("--benchmark-images must be positive")
    model, config, device = load_model(args.checkpoint, args.device, args.backbone_repo)
    root = project_path(args.data_root or config["data_root"])
    dataset = ComponentDataset(root, "val", use_depth=config["use_depth"], limit=args.limit)
    loader = make_loader(dataset, workers=args.workers)
    predictions = collect_predictions(model, loader, device)
    metrics = evaluate_predictions(dataset.build_coco(), predictions, dataset.ids)
    metrics["images"] = len(dataset)
    metrics["image_size"] = config["image_size"]
    metrics["checkpoint"] = str(project_path(args.checkpoint))
    output = project_path(args.output)
    write_json(output / "metrics.json", metrics)
    write_json(output / "val_predictions.json", predictions, indent=None)
    if args.benchmark:
        report = benchmark_model(model, dataset, device, args.benchmark_images)
        write_json(output / "benchmark.json", report)
    print(f"Mask AP_s={metrics['segm']['AP_s']:.4f}; Mask AP={metrics['segm']['AP']:.4f}; images={len(dataset)}")


if __name__ == "__main__":
    main()
