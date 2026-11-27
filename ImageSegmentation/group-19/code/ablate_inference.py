"""在 val 等间隔子集上对比不同推理配置（尺度、翻转、多趟合并）的指标。

用于决定最终提交采用哪种推理方案；所有配置共用同一批图像与同一 COCOeval
协议，结果直接可比。用法：
    python ablate_inference.py --checkpoint /data1/.../best.pth --stride 6
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2f_rgbd.data import RGBDInstanceDataset, collate_targets
from m2f_rgbd.engine import collect_predictions_ms_tta, evaluate_predictions
from m2f_rgbd.utils import make_loader, project_path, setup_device, write_json
from predict_m2f import load_model_m2f

CONFIGS = (
    ("scale1.0", {"scales": (1.0,), "flip": False}),
    ("scale1.0+flip", {"scales": (1.0,), "flip": True}),
    ("scale1.25", {"scales": (1.25,), "flip": False}),
    ("scale1.25+flip", {"scales": (1.25,), "flip": True}),
    ("ms1.0+1.25+flip", {"scales": (1.0, 1.25), "flip": True}),
)


def main():
    """逐配置推理并评价，打印并保存指标表；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stride", type=int, default=6, help="等间隔抽取 val 图像")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scales", default="", help="逗号分隔尺度；给出时只跑这些单尺度配置")
    parser.add_argument("--with-flip", action="store_true", help="--scales 模式下同时测翻转合并")
    parser.add_argument("--output", default="outputs/ablation_inference")
    args = parser.parse_args()
    if args.scales:
        scales = [float(item) for item in args.scales.split(",") if item.strip()]
        configs = [(f"scale{scale:g}", {"scales": (scale,), "flip": False}) for scale in scales]
        if args.with_flip:
            configs += [(f"scale{scale:g}+flip", {"scales": (scale,), "flip": True})
                        for scale in scales]
    else:
        configs = CONFIGS

    model, config, device = load_model_m2f(args.checkpoint, args.device)
    root = project_path(args.data_root or config["data_root"])
    dataset = RGBDInstanceDataset(root, "val", mode=config["mode"], jitter=None, crop_size=None)
    indices = list(range(0, len(dataset), max(1, args.stride)))
    subset = Subset(dataset, indices)
    image_ids = [dataset.ids[index] for index in indices]
    ground_truth = dataset.build_coco()
    results = {}
    for name, kwargs in configs:
        loader = make_loader(subset, args.batch_size, args.workers, collate_fn=collate_targets)
        records = collect_predictions_ms_tta(model, loader, device, config["score_threshold"],
                                             amp_dtype=config["amp_dtype"],
                                             scales=kwargs["scales"], flip=kwargs["flip"])
        metrics = evaluate_predictions(ground_truth, records, image_ids)
        segm = metrics["segm"]
        results[name] = {"AP": segm["AP"], "AP50": segm["AP50"], "AP75": segm["AP75"],
                         "AP_s": segm["AP_s"], "AP_m": segm["AP_m"], "AP_l": segm["AP_l"],
                         "detections": len(records)}
        print(f"{name:18s} AP={segm['AP']:.4f} AP50={segm['AP50']:.4f} "
              f"AP75={segm['AP75']:.4f} AP_s={segm['AP_s']:.4f} "
              f"AP_m={segm['AP_m']:.4f} dets={len(records)}", flush=True)
        del loader, records
        torch.cuda.empty_cache()
    output = project_path(args.output)
    write_json(output / "ablation.json", {"images": len(indices), "results": results})
    print(f"Saved {output / 'ablation.json'}")


if __name__ == "__main__":
    main()
