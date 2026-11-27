"""用训练好的 Mask2Former 对 test 推理并导出课程提交文件（掩码 JSON + CSV）。

复用旧管线 rgbdseg.predict.export_submission 保证 CSV/JSON 格式与助教模板一致。
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2f_rgbd.data import RGBDInstanceDataset, collate_targets
from m2f_rgbd.engine import collect_predictions, collect_predictions_ms_tta
from m2f_rgbd.model import build_mask2former
from m2f_rgbd.utils import load_checkpoint, make_loader, project_path, setup_device, write_json
from predict import export_submission


def load_model_m2f(checkpoint_path, requested_device="auto"):
    """读取整模型 checkpoint，返回 (model, config, device)。"""
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"].copy()
    device = setup_device(requested_device, config["seed"])
    model = build_mask2former(
        project_path(config["checkpoint_dir"]),
        num_channels=config["num_channels"],
        num_labels=config["num_labels"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, device


def main():
    """对测试图像推理并导出两种提交文件及来源记录；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--template", default="submission.csv", help="Path relative to data root, or absolute")
    parser.add_argument("--limit", type=int, default=0, help="N>0 writes preview files, not a complete submission")
    parser.add_argument("--score-mode", choices=["legacy_bf16", "class_fp32", "mask_quality"],
                        default="class_fp32", help="Instance ranking; use mask_quality for Meta's mask-aware score")
    parser.add_argument("--tta", action="store_true", help="Horizontal-flip test-time augmentation")
    parser.add_argument("--tta-scales", default="1.0,1.125",
                        help="Comma-separated scales for multi-scale TTA (with --tta)")
    parser.add_argument("--no-flip", action="store_true", help="--tta 时只做多尺度，不做水平翻转")
    parser.add_argument("--output", default="outputs/submission_m2f")
    args = parser.parse_args()
    if args.tta and args.score_mode != "class_fp32":
        parser.error("Selected score mode is currently validated for single-pass inference only; omit --tta")
    if not 0.001 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0.001 and 1")
    model, config, device = load_model_m2f(args.checkpoint, args.device)
    root = project_path(args.data_root or config["data_root"])
    dataset = RGBDInstanceDataset(root, "test", mode=config["mode"],
                                  jitter=None, crop_size=None, limit=args.limit)
    loader = make_loader(dataset, args.batch_size, args.workers, collate_fn=collate_targets)
    if args.tta:
        scales = tuple(float(scale) for scale in args.tta_scales.split(",") if scale.strip())
        records = collect_predictions_ms_tta(model, loader, device, args.score_threshold,
                                             amp_dtype=config["amp_dtype"], scales=scales,
                                             flip=not args.no_flip)
    else:
        records = collect_predictions(model, loader, device, args.score_threshold,
                                      amp_dtype=config["amp_dtype"], score_mode=args.score_mode)
    output = project_path(args.output)
    summary = export_submission(records, dataset.ids, root / args.template, output,
                                csv_format="masks", partial=bool(args.limit))
    summary.update(checkpoint=str(project_path(args.checkpoint)),
                   score_threshold=args.score_threshold, score_mode=args.score_mode,
                   postprocess_precision="float32" if args.score_mode != "legacy_bf16" else "bfloat16",
                   model=config["checkpoint_dir"], mode=config["mode"],
                   tta=("ms" if args.no_flip else "ms+flip") if args.tta else None,
                   tta_scales=args.tta_scales if args.tta else None,
                   depth_normalization="per-image relative depth" if config["mode"] == "rgbd" else None)
    write_json(output / f"{summary['basename']}_info.json", summary)
    print(f"Saved {output}; images={summary['images']}; instances={summary['instances']}; "
          f"empty_masks_removed={summary['empty_masks_removed']}; "
          f"images_without_predictions={summary['images_without_predictions']}; partial={summary['partial']}")


if __name__ == "__main__":
    main()
