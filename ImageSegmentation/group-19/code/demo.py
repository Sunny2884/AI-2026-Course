"""RGB-D 实例分割 Demo：并排展示 RGB、有效深度、GT 与两个模型的预测。

用法：
    python demo.py --image-ids 201960001 201960003 \\
        --baseline-checkpoint outputs/dinov3_rgbd/best.pth \\
        --improved-checkpoint checkpoints/m2f_finetune_best.pth

不传 --image-ids 时自动选取 val 中小目标较多（且含典型失败模式）的图像。
输出 PNG 保存到 --output 目录，每张图 5 列：RGB、深度、GT、基线、改进模型。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rgbdseg.data import ComponentDataset
from rgbdseg.utils import load_model as load_baseline_model
from m2f_rgbd.data import RGBDInstanceDataset, decode_mask
from m2f_rgbd.utils import project_path, setup_device
from predict_m2f import load_model_m2f


def instance_colors(count):
    """为 count 个实例生成高区分度的 BGR 颜色列表（HSV 表示）。"""
    colors = []
    for index in range(count):
        hue = int(180 * (index * 0.61803398875) % 180)  # 黄金角，相邻实例颜色差异大
        value = 170 + 85 * ((index // 12) % 2)
        colors.append((hue, 160, value))
    return colors


def overlay_masks(rgb, masks, alpha=0.55):
    """把 uint8[N,H,W] 掩码叠加到 RGB 上，返回 BGR uint8 图像。"""
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    hsv_colors = instance_colors(len(masks))
    for mask, hsv in zip(masks, hsv_colors):
        bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        region = mask.astype(bool)
        canvas[region] = alpha * bgr + (1 - alpha) * canvas[region]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(round(c)) for c in bgr), 1)
    return canvas.astype(np.uint8)


def depth_visual(depth_relative):
    """把相对深度 [0,1] 渲染成彩色图；无效区域（0）显示为黑色。"""
    valid = depth_relative > 0
    shown = np.clip(depth_relative / max(depth_relative.max(), 1e-6), 0, 1)
    hsv = np.zeros((*depth_relative.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (120 * (1 - shown)).astype(np.uint8)  # 近红远蓝
    hsv[..., 1] = 255
    hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def load_rgb_depth(root, info):
    """读取 RGB 与相对深度；返回 (rgb uint8, relative float32)。"""
    with Image.open(root / info["file_name"]) as source:
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
    with Image.open(root / info["depth_file_name"]) as source:
        raw = np.array(source, dtype=np.float32)
    valid = raw != int(info["depth_invalid_code"])
    relative = np.zeros_like(raw)
    if valid.any():
        maximum = float(raw[valid].max())
        if maximum > 0:
            relative[valid] = raw[valid] / maximum
    return rgb, relative


def baseline_predict(model, image_tensor, device, score_threshold=0.05):
    """用旧管线模型预测；返回 uint8[N,1024,1024] 掩码列表。"""
    with torch.inference_mode():
        output = model([image_tensor.to(device)])[0]
    masks = output["masks"][:, 0]
    scores = output["scores"]
    keep = (scores >= score_threshold) & (masks.flatten(1).max(1).values > 0.5)
    return (masks[keep] >= 0.5).cpu().numpy().astype(np.uint8)


def m2f_predict(model, target, device, score_threshold=0.05):
    """用 Mask2Former 模型预测；返回 uint8[N,1024,1024] 掩码列表。"""
    from m2f_rgbd.engine import outputs_to_records
    with torch.inference_mode():
        outputs = model(
            pixel_values=target["pixel_values"][None].to(device),
            pixel_mask=target["pixel_mask"][None].to(device),
        )
    records = outputs_to_records(outputs, [int(target["image_id"])], score_threshold, 0.5, 1024)
    from pycocotools import mask as mask_utils
    masks = []
    for record in records:
        decoded = mask_utils.decode(record["segmentation"])
        masks.append(decoded.astype(np.uint8))
    return masks


def pick_demo_images(root, count=6):
    """自动挑选 val 中小目标较多的图像 id 列表。"""
    dataset = RGBDInstanceDataset(root, "val", mode="rgbd", jitter=None, crop_size=None)
    candidates = []
    for image_id in dataset.ids:
        items = dataset._annotations_for(image_id)
        n_small = sum(item["area"] < 1024 for item in items)
        candidates.append((n_small, image_id))
    candidates.sort(reverse=True)
    return [image_id for _, image_id in candidates[:count]]


def main():
    """生成并保存 Demo 图像；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ids", type=int, nargs="*", help="val 图像 id；缺省自动选择小目标丰富的图像")
    parser.add_argument("--baseline-checkpoint", default="outputs/baseline/best.pth")
    parser.add_argument("--improved-checkpoint", required=True)
    parser.add_argument("--data-root", default="./dataset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/demo")
    args = parser.parse_args()

    root = project_path(args.data_root)
    device = setup_device(args.device, 42)
    baseline_model, baseline_config, _ = load_baseline_model(args.baseline_checkpoint, args.device)
    improved_model, improved_config, _ = load_model_m2f(args.improved_checkpoint, args.device)

    val_meta = ComponentDataset(root, "val", use_depth=True, limit=0)
    info_by_id = {int(image["id"]): image for image in val_meta.images}
    ids = args.image_ids or pick_demo_images(root)
    ids = [image_id for image_id in ids if image_id in info_by_id]
    if not ids:
        raise SystemExit("没有可用的演示图像 id")

    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    m2f_dataset = RGBDInstanceDataset(root, "val", mode="rgbd", jitter=None, crop_size=None)

    for image_id in ids:
        info = info_by_id[image_id]
        rgb, relative = load_rgb_depth(root, info)
        gt_items = [item for item in m2f_dataset._annotations_for(image_id) if item["area"] >= 1]
        gt_masks = [decode_mask(item["segmentation"], 1024, 1024) for item in gt_items]

        image_tensor, _ = ComponentDataset(root, "val", use_depth=True, limit=0)[
            val_meta.ids.index(image_id)]
        target = m2f_dataset[m2f_dataset.ids.index(image_id)]

        base_masks = baseline_predict(baseline_model, image_tensor, device, args.score_threshold)
        improved_masks = m2f_predict(improved_model, target, device, args.score_threshold)

        panels = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth_visual(relative),
            overlay_masks(rgb, gt_masks) if gt_masks else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            overlay_masks(rgb, base_masks) if len(base_masks) else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            overlay_masks(rgb, improved_masks) if len(improved_masks) else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        ]
        titles = ["RGB", "Depth", f"GT ({len(gt_masks)})",
                  f"Baseline ({len(base_masks)})", f"Improved ({len(improved_masks)})"]
        for panel, title in zip(panels, titles):
            cv2.putText(panel, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        grid = np.concatenate(panels, axis=1)
        path = output / f"demo_{image_id}.png"
        cv2.imwrite(str(path), grid)
        print(f"saved {path}  GT={len(gt_masks)} baseline={len(base_masks)} improved={len(improved_masks)}")


if __name__ == "__main__":
    main()
