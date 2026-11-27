"""Mask2Former 管线数据集：RGB-D 四通道、LSJ 大尺度抖动、小目标锚定裁剪。

与 rgbdseg.data.ComponentDataset 使用相同的数据目录与标注分片，但：
- 训练增强采用 Mask2Former 官方的 large-scale jitter（缩放抖动 [0.1, 2.0]、
  随机 1024 裁剪、水平翻转），并可选“小目标锚定裁剪”；
- 返回 Hugging Face Mask2Former 所需的 pixel_values / pixel_mask /
  class_labels / mask_labels 结构（单类时 class 标签为 0，背景由 no-object
  查询承担，num_labels=1 由模型配置决定）；
- 几何变换严格同步 RGB、深度与实例掩码：RGB 用双线性插值，深度与掩码用
  最近邻，深度无效区域在缩放后仍保持 0。
"""
from __future__ import annotations

import gzip
import json
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_json(path: Path):
    """读取 JSON 或 JSON.GZ；path 为文件路径，返回解析后的字典或列表。"""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def decode_mask(segmentation, height: int, width: int) -> np.ndarray:
    """将 COCO RLE/多边形 segmentation 解码为 uint8[height,width] 二值掩码。"""
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    else:
        rle = dict(segmentation)
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=np.uint8)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """从 uint8[N,H,W] 掩码计算 [N,4] 边界框；返回 float32 数组。"""
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for i in range(masks.shape[0]):
        ys, xs = np.nonzero(masks[i])
        if len(xs) == 0:
            boxes[i] = [0.0, 0.0, 0.0, 0.0]
        else:
            boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes


class RGBDInstanceDataset(torch.utils.data.Dataset):
    """为 Mask2Former 提供 RGB-D 输入与实例标签的课程数据集。

    参数：
        root：数据根目录（含 rgb/depth/annotations）；
        split：train/val/test；
        mode：rgbd 返回 RGB+相对深度四通道；rgb 仅返回 RGB 三通道；
        jitter：训练时的 LSJ 缩放抖动区间 (lo, hi)，None 表示不缩放；
        crop_size：LSJ 裁剪尺寸，None 表示整图；
        small_crop_prob：以“小目标锚定”方式采样裁剪位置的概率（仅训练）；
        flip_prob：水平翻转概率（仅训练）；
        limit：只用前 N 张图；0 表示全部。
    """

    def __init__(
        self,
        root,
        split: str,
        mode: str = "rgbd",
        jitter=(0.1, 2.0),
        crop_size: int | None = 1024,
        small_crop_prob: float = 0.5,
        flip_prob: float = 0.5,
        limit: int | None = 0,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须是 train、val 或 test")
        if mode not in {"rgb", "rgbd"}:
            raise ValueError("mode 必须是 rgb 或 rgbd")
        if limit == 0:
            limit = None
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.jitter = jitter
        self.crop_size = crop_size
        self.small_crop_prob = small_crop_prob if split == "train" else 0.0
        self.flip_prob = flip_prob if split == "train" else 0.0
        self.augment = split == "train"
        self.images: list[dict] = []
        self.shard_files: list[Path] = []
        self._shard_by_id: dict[int, Path] = {}
        # 每图标注缓存（按 image_id），训练随机访问时避免反复解压分片。
        self._ann_cache: OrderedDict[int, list[dict]] = OrderedDict()
        self._cache_limit = 4096

        if split in {"train", "val"}:
            files = sorted((self.root / "annotations" / split).glob("*.json.gz"))
            for path in files:
                records = sorted(read_json(path)["images"], key=lambda item: int(item["id"]))
                self.shard_files.append(path)
                for record in records:
                    self.images.append(self._image_info(record))
                    self._shard_by_id[int(record["id"])] = path
                    if limit is not None and len(self.images) >= limit:
                        break
                if limit is not None and len(self.images) >= limit:
                    break
        else:
            metadata = self.root / "test_images.json"
            if metadata.is_file():
                data = read_json(metadata)
                records = data["images"] if isinstance(data, dict) else data
                records = sorted(records, key=lambda item: int(item["id"]))
                self.images = [self._image_info(record) for record in records[:limit]]
            else:
                files = sorted((self.root / "rgb" / "test").glob("*.png"))[:limit]
                for path in files:
                    scene, view = path.stem.split("_v")
                    self.images.append({
                        "id": int(scene) * 1000 + int(view) + 1,
                        "file_name": path.relative_to(self.root).as_posix(),
                        "depth_file_name": f"depth/test/{path.name}",
                        "height": 1024, "width": 1024, "depth_invalid_code": 0,
                    })
        self.images.sort(key=lambda item: int(item["id"]))
        self.ids = [int(image["id"]) for image in self.images]
        if not self.ids:
            raise FileNotFoundError(f"{self.root} 下未找到 {split} 数据，请检查数据目录")
        self._selected_ids = set(self.ids)

    def _image_info(self, record: dict) -> dict:
        """从一条官方图像 record 提取必要字段，返回路径相对 root 的元数据字典。"""
        fields = (
            "id", "file_name", "depth_file_name", "height", "width",
            "depth_invalid_code", "depth_scale_m", "scene_id",
        )
        image = {key: record[key] for key in fields if key in record}
        image["id"] = int(image["id"])
        filename = Path(image["file_name"])
        if len(filename.parts) == 1:
            image["file_name"] = f"rgb/{self.split}/{filename.name}"
        image.setdefault("depth_file_name", f"depth/{self.split}/{filename.name}")
        return image

    def __len__(self) -> int:
        """无需参数；返回当前划分经过 limit 限制后的图像数量。"""
        return len(self.images)

    def _annotations_for(self, image_id: int) -> list[dict]:
        """读取 image_id 的实例标注；最近使用的图像标注缓存在内存中。"""
        if image_id in self._ann_cache:
            self._ann_cache.move_to_end(image_id)
            return self._ann_cache[image_id]
        path = self._shard_by_id[image_id]
        items = []
        for annotation in read_json(path)["annotations"]:
            if int(annotation["image_id"]) == image_id and not annotation.get("iscrowd", 0):
                items.append({
                    key: annotation[key]
                    for key in ("bbox", "area", "segmentation")
                    if key in annotation
                })
        self._ann_cache[image_id] = items
        if len(self._ann_cache) > self._cache_limit:
            self._ann_cache.popitem(last=False)
        return items

    def _load_rgbd(self, info: dict) -> tuple[np.ndarray, np.ndarray]:
        """读取 RGB[1024,1024,3] 与相对深度[1024,1024]；无效深度置 0。"""
        with Image.open(self.root / info["file_name"]) as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        if self.mode == "rgb":
            return rgb, None
        with Image.open(self.root / info["depth_file_name"]) as source:
            raw = np.array(source, dtype=np.float32)
        valid = raw != int(info["depth_invalid_code"])
        relative = np.zeros_like(raw)
        if valid.any():
            maximum = float(raw[valid].max())
            if maximum > 0:
                relative[valid] = raw[valid] / maximum
        return rgb, relative

    def _sample_crop(self, height: int, width: int, small_boxes: list[tuple]) -> tuple[int, int]:
        """在缩放后的图上选择 crop_size 裁剪的左上角；返回 (y0, x0)。

        以 small_crop_prob 的概率把裁剪锚定在某个小目标附近（保证其完整进入
        裁剪窗），否则均匀随机。小目标锚定提高小实例在训练中的有效分辨率。
        """
        crop = self.crop_size
        if small_boxes and random.random() < self.small_crop_prob:
            x1, y1, x2, y2 = random.choice(small_boxes)
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            x0 = random.uniform(x2 - crop, x1) if crop < width else 0
            y0 = random.uniform(y2 - crop, y1) if crop < height else 0
            x0 = float(np.clip(x0, 0, max(width - crop, 0)))
            y0 = float(np.clip(y0, 0, max(height - crop, 0)))
            return int(round(y0)), int(round(x0))
        y0 = random.randrange(max(height - crop, 0) + 1) if crop < height else 0
        x0 = random.randrange(max(width - crop, 0) + 1) if crop < width else 0
        return y0, x0

    def __getitem__(self, index: int):
        """读取第 index 张图；返回模型输入字典和 image_id。"""
        info = self.images[index]
        rgb, depth = self._load_rgbd(info)
        height, width = rgb.shape[:2]
        mask_labels: list[np.ndarray] = []

        if self.split == "train":
            for annotation in self._annotations_for(info["id"]):
                decoded = decode_mask(annotation["segmentation"], height, width)
                if decoded.any():
                    mask_labels.append(decoded)

        # ---- LSJ：缩放抖动 → 裁剪 → 翻转，RGB/深度/掩码严格同步 ----
        if self.augment and self.jitter is not None:
            lo, hi = self.jitter
            scale = lo + (hi - lo) * random.random()
            resized = max(int(round(height * scale)), 1)
            if resized != height:
                rgb = cv2.resize(rgb, (resized, resized), interpolation=cv2.INTER_LINEAR)
                if depth is not None:
                    depth = cv2.resize(depth, (resized, resized), interpolation=cv2.INTER_NEAREST)
                mask_labels = [
                    cv2.resize(mask, (resized, resized), interpolation=cv2.INTER_NEAREST)
                    for mask in mask_labels
                ]
                height = width = resized

        if self.augment and self.crop_size is not None and height > self.crop_size:
            boxes = masks_to_boxes(np.stack(mask_labels)) if mask_labels else np.zeros((0, 4))
            small_boxes = [
                tuple(box) for box, mask in zip(boxes, mask_labels) if mask.sum() < 1024
            ]
            y0, x0 = self._sample_crop(height, width, small_boxes)
            crop = self.crop_size
            rgb = rgb[y0:y0 + crop, x0:x0 + crop]
            if depth is not None:
                depth = depth[y0:y0 + crop, x0:x0 + crop]
            mask_labels = [mask[y0:y0 + crop, x0:x0 + crop] for mask in mask_labels]
            height = width = crop

        pixel_mask = np.ones((height, width), dtype=np.float32)
        if self.augment and self.crop_size is not None and height < self.crop_size:
            # 缩放后小于裁剪尺寸：四周补零，pixel_mask 标出无效区域。
            pad = self.crop_size - height
            top, left = pad // 2, pad // 2
            bottom, right = pad - top, pad - left
            rgb = np.pad(rgb, ((top, bottom), (left, right), (0, 0)))
            pixel_mask = np.pad(pixel_mask, ((top, bottom), (left, right)))
            if depth is not None:
                depth = np.pad(depth, ((top, bottom), (left, right)))
            mask_labels = [np.pad(mask, ((top, bottom), (left, right))) for mask in mask_labels]
            height = width = self.crop_size

        if self.augment and random.random() < self.flip_prob:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            if depth is not None:
                depth = np.ascontiguousarray(depth[:, ::-1])
            mask_labels = [np.ascontiguousarray(mask[:, ::-1]) for mask in mask_labels]
            pixel_mask = np.ascontiguousarray(pixel_mask[:, ::-1])

        # ---- 归一化与张量化 ----
        pixels = rgb.astype(np.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        channels = [pixels]
        if depth is not None:
            channels.append(depth[..., None])
        pixel_values = torch.from_numpy(np.concatenate(channels, axis=2).transpose(2, 0, 1))

        # 掩码统一 float32：HF 损失中的 grid_sample 不支持 Byte 输入。
        masks_out = [torch.from_numpy(mask.astype(np.float32)) for mask in mask_labels if mask.any()]
        if masks_out:
            masks_tensor = torch.stack(masks_out)
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.float32)

        target = {
            "pixel_values": pixel_values.contiguous(),
            "pixel_mask": torch.from_numpy(pixel_mask),
            "class_labels": torch.zeros(masks_tensor.shape[0], dtype=torch.int64),
            "mask_labels": masks_tensor,
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
        }
        return target

    def build_coco(self):
        """无需参数；构建 val 评价 COCO 对象（与旧管线同一协议），test 不支持。"""
        from pycocotools.coco import COCO

        if self.split == "test":
            raise ValueError("test 没有公开标注，不能构建评价真实标签")
        fields = ("id", "image_id", "bbox", "area", "iscrowd", "category_id", "segmentation")
        annotations = []
        for path in self.shard_files:
            for annotation in read_json(path)["annotations"]:
                if int(annotation["image_id"]) in self._selected_ids:
                    item = {key: annotation[key] for key in fields if key in annotation}
                    item.setdefault("iscrowd", 0)
                    annotations.append(item)
        coco = COCO()
        coco.dataset = {
            "info": {},
            "images": [
                {key: image[key] for key in ("id", "file_name", "height", "width")}
                for image in self.images
            ],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "component"}],
        }
        coco.createIndex()
        return coco


def collate_targets(batch):
    """将 batch 内的 target 字典合并为模型输入；返回 (pixel_values, pixel_mask, class_labels, mask_labels, image_ids)。

    pixel_values/pixel_mask 直接堆叠；class_labels/mask_labels 是每图变长的
    列表（每项为 [N_i] / [N_i,H,W] 张量），按原样传给模型（与 HF Mask2Former
    的前向接口一致）。
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    class_labels = [item["class_labels"] for item in batch]
    mask_labels = [item["mask_labels"] for item in batch]
    image_ids = [int(item["image_id"]) for item in batch]
    return pixel_values, pixel_mask, class_labels, mask_labels, image_ids
