"""读取课程数据集的 RGB、相对深度及 COCO 实例标注。"""
from __future__ import annotations

import gzip
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset


ANNOTATION_FIELDS = (
    "id", "image_id", "bbox", "area", "iscrowd", "category_id", "segmentation",
)


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


class ComponentDataset(Dataset):
    """课程元器件数据集，逐图读取输入并按需解码训练掩码。

    参数：root 为含 rgb/depth/annotations 的数据根目录；split 为 train/val/test；
    use_depth 决定返回 RGB 三通道还是 RGB+相对深度四通道；augment 仅对 train
    启用概率为 0.5 的水平翻转；limit 限制前 N 张图，None 或 0 表示全部。
    depth_invalid_code 用于 test 元数据缺少无效值编码时的显式假设。

    返回：dataset[index] 为 (float32[C,H,W] 输入, target 字典)。train 的 target
    含框、类别、掩码等标签；val/test 的 target 仅含 image_id。评价标签由
    build_coco() 单独读取，避免验证推理时重复解码真实掩码。
    """

    def __init__(
        self,
        root: Path,
        split: str,
        use_depth: bool = True,
        augment: bool = False,
        limit: int | None = None,
        depth_invalid_code: int | None = None,
    ):
        """按类说明的参数建立图像索引；仅保存必要元数据，不预加载像素和掩码。"""
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须是 train、val 或 test")
        if limit == 0:
            limit = None
        if limit is not None and limit < 0:
            raise ValueError("limit 必须为非负整数，或使用 None 读取全部图像")
        self.root = Path(root)
        self.split = split
        self.use_depth = use_depth
        self.augment = augment and split == "train"
        self.images: list[dict] = []
        self.shard_files: list[Path] = []
        self._shard_by_id: dict[int, Path] = {}
        self._cache: OrderedDict[Path, dict[int, list[dict]]] = OrderedDict()

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
                if use_depth and depth_invalid_code is None:
                    raise ValueError(
                        "缺少 test_images.json。使用深度输入时，请显式传入 "
                        "--depth-invalid-code 0，表示假设 0 为无效深度；"
                        "该假设未经官方 test 元数据确认。相对深度无需米制尺度。"
                    )
                # 文件名 0202310_v000.png 对应 scene*1000+view+1；提交前另与模板核对。
                files = sorted((self.root / "rgb" / "test").glob("*.png"))[:limit]
                for path in files:
                    scene, view = path.stem.split("_v")
                    with Image.open(path) as source:
                        width, height = source.size
                    self.images.append({
                        "id": int(scene) * 1000 + int(view) + 1,
                        "file_name": path.relative_to(self.root).as_posix(),
                        "depth_file_name": f"depth/test/{path.name}",
                        "height": height,
                        "width": width,
                    })

        self.images.sort(key=lambda item: int(item["id"]))
        self.ids = [int(image["id"]) for image in self.images]
        if not self.ids:
            raise FileNotFoundError(f"{self.root} 下未找到 {split} 数据，请检查数据目录")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"{split} 图像 ID 重复")
        self._selected_ids = set(self.ids)
        if use_depth:
            for image in self.images:
                if "depth_invalid_code" not in image:
                    if split == "test" and depth_invalid_code is not None:
                        image["depth_invalid_code"] = int(depth_invalid_code)
                    else:
                        raise ValueError(
                            f"{split} 图像缺少 depth_invalid_code。test 元数据缺失时，"
                            "可显式传入 --depth-invalid-code 0，表示假设 0 为无效深度；"
                            "该假设未经官方 test 元数据确认。相对深度无需米制尺度。"
                        )

    def _image_info(self, record: dict) -> dict:
        """从一条官方图像 record 提取必要字段，返回路径相对 root 的元数据字典。"""
        fields = (
            "id", "file_name", "depth_file_name", "height", "width", "depth_invalid_code", "scene_id",
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
        """读取 image_id 的实例标注并返回列表；每个进程最多缓存最近 8 个分片。"""
        path = self._shard_by_id[image_id]
        if path not in self._cache:
            by_image = defaultdict(list)
            for annotation in read_json(path)["annotations"]:
                if int(annotation["image_id"]) in self._selected_ids:
                    item = {key: annotation[key] for key in ANNOTATION_FIELDS if key in annotation}
                    by_image[int(item["image_id"])].append(item)
            self._cache[path] = dict(by_image)
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        return self._cache[path].get(image_id, [])

    def __getitem__(self, index: int):
        """读取第 index 张图；返回输入张量和训练标签，val/test 仅返回 image_id 标签。"""
        info = self.images[index]
        with Image.open(self.root / info["file_name"]) as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if self.use_depth:
            with Image.open(self.root / info["depth_file_name"]) as source:
                raw = np.array(source, dtype=np.float32)
            valid = raw != int(info["depth_invalid_code"])
            relative_depth = np.zeros_like(raw)
            # 同一图内用最大有效值归一化：raw*scale / max(raw*scale) = raw/max(raw)。
            if valid.any():
                maximum = float(raw[valid].max())
                if maximum > 0:
                    relative_depth[valid] = raw[valid] / maximum
            image = torch.cat([image, torch.from_numpy(relative_depth).unsqueeze(0)], dim=0)

        target = {"image_id": torch.tensor(info["id"], dtype=torch.int64)}
        if self.split == "train":
            boxes, masks, areas = [], [], []
            for annotation in self._annotations_for(info["id"]):
                if annotation.get("iscrowd", 0):
                    continue
                x, y, box_width, box_height = map(float, annotation["bbox"])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
                if x2 <= x1 or y2 <= y1:
                    continue
                mask = decode_mask(annotation["segmentation"], height, width)
                if not mask.any():
                    continue
                boxes.append([x1, y1, x2, y2])
                masks.append(mask)
                areas.append(float(annotation["area"]))
            target.update({
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.ones(len(boxes), dtype=torch.int64),
                "masks": torch.from_numpy(
                    np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
                ),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            })
            if self.augment and random.random() < 0.5:
                image = image.flip(-1)
                target["masks"] = target["masks"].flip(-1)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image.contiguous(), target

    def build_coco(self) -> COCO:
        """无需参数；读取所选 train/val 标注构建 COCO 评价对象，test 不提供标签。"""
        if self.split == "test":
            raise ValueError("test 没有公开标注，不能构建评价真实标签")
        annotations = []
        for path in self.shard_files:
            for annotation in read_json(path)["annotations"]:
                if int(annotation["image_id"]) in self._selected_ids:
                    item = {key: annotation[key] for key in ANNOTATION_FIELDS if key in annotation}
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


def collate_fn(batch):
    """将 batch 内的 (image,target) 样本转为两个列表，保留每图不同的实例数量。"""
    images, targets = zip(*batch)
    return list(images), list(targets)
