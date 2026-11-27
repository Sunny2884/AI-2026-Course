"""构建 4 通道 RGB-D Mask2Former（Swin 主干）并加载 COCO 实例分割预训练权重。

主干 stem 的 RGB 三通道沿用预训练卷积参数，新增深度通道以 RGB 参数均值
初始化，与旧管线 DINOv3 RGB-D 融合的初始化方式保持一致；其余层直接加载
facebook/mask2former-swin-*-coco-instance 的官方权重。分类头因类别数不同
（80 类 COCO -> 单类）从头训练。
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


def _copy_stem_weights(model: Mask2FormerForUniversalSegmentation, checkpoint_dir: str):
    """把 3 通道预训练 stem 参数复制进 4 通道 stem；深度通道取 RGB 均值。"""
    state = torch.load(Path(checkpoint_dir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    prefix = "model.pixel_level_module.encoder.embeddings.patch_embeddings"
    source = state[f"{prefix}.projection.weight"]
    stem = model.model.pixel_level_module.encoder.swin.embeddings.patch_embeddings.projection
    if stem.weight.shape[1] != 4:
        raise ValueError(f"expected 4-channel stem, got {tuple(stem.weight.shape)}")
    with torch.no_grad():
        stem.weight[:, :3].copy_(source)
        stem.weight[:, 3:4].copy_(source.mean(dim=1, keepdim=True))
        if stem.bias is not None:
            stem.bias.copy_(state[f"{prefix}.projection.bias"])
    if not torch.equal(stem.weight[:, :3], source):
        raise RuntimeError("stem RGB 通道复制失败，模型主干未继承预训练权重")
    return model


def build_mask2former(
    checkpoint_dir: str,
    num_channels: int = 4,
    num_labels: int = 1,
    local_files_only: bool = True,
):
    """加载预训练 Mask2Former 并改造为 num_channels 输入的 RGB-D 模型。

    参数：
        checkpoint_dir：已下载的 facebook/mask2former-swin-*-coco-instance 目录；
        num_channels：输入通道数，4 为 RGB+相对深度；
        num_labels：前景类别数（单类任务为 1，no-object 查询由模型内部处理）。
    返回：训练模式的 Mask2FormerForUniversalSegmentation。
    """
    config = Mask2FormerConfig.from_pretrained(checkpoint_dir, local_files_only=local_files_only)
    backbone_config = config.backbone_config
    if backbone_config is None or getattr(backbone_config, "model_type", None) != "swin":
        raise ValueError("当前实现只支持 Swin 主干的 Mask2Former checkpoint")
    backbone_config.num_channels = num_channels
    config.backbone_config = backbone_config
    config.num_labels = num_labels
    # 分类头与 stem 形状与预训练不同，其余层严格加载。
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint_dir, config=config, ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    if num_channels != 3:
        model = _copy_stem_weights(model, checkpoint_dir)
    # Mask2Former 只消费各 stage 的 feature_maps，Swin 末端 layernorm 不参与
    # 损失；冻结它避免 DDP 因“未使用参数”报错（find_unused_parameters=False）。
    model.model.pixel_level_module.encoder.swin.layernorm.requires_grad_(False)
    return model
