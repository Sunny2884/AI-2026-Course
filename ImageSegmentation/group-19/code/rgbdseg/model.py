"""DINOv3 ConvNeXt-Tiny + RGB-D 早期融合 + FPN + Mask R-CNN。"""

from collections import OrderedDict
from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class DinoBackbone(nn.Module):
    """提取 DINOv3 原始四尺度特征，再用 FPN 统一为 256 通道。

    参数：config 指定源码、权重与深度开关；pretrained 决定是否加载官方权重。
    返回：DinoBackbone 实例，其 forward 返回五种尺度的 FPN 特征字典。
    """

    def __init__(self, config: dict, pretrained: bool = False):
        """参数：config 为项目配置；pretrained 决定是否加载本地官方权重。

        返回：None；初始化后 out_channels 为 Mask R-CNN 所需的通道数。
        """
        super().__init__()
        project_root = Path(__file__).resolve().parent.parent
        repo = project_root / config["backbone_repo"]
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"缺少 DINOv3 源码：{repo}，请按 README 准备源码。")
        # 直接导入骨干，避免 hubconf 导入本项目不用的官方下游任务及其依赖。
        sys.path.insert(0, str(repo))
        from dinov3.hub.backbones import dinov3_convnext_tiny

        self.encoder = dinov3_convnext_tiny(pretrained=False)
        if pretrained:
            weights = project_root / config["backbone_weights"]
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        else:
            # 恢复完整 checkpoint 前先统一初始化，随后由 checkpoint 覆盖参数。
            self.encoder.init_weights()

        if config["use_depth"]:
            rgb_conv = self.encoder.downsample_layers[0][0]
            rgbd_conv = nn.Conv2d(
                4, rgb_conv.out_channels, kernel_size=rgb_conv.kernel_size,
                stride=rgb_conv.stride, padding=rgb_conv.padding,
                bias=rgb_conv.bias is not None,
            )
            with torch.no_grad():
                rgbd_conv.weight[:, :3].copy_(rgb_conv.weight)
                rgbd_conv.weight[:, 3:4].copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
                if rgb_conv.bias is not None:
                    rgbd_conv.bias.copy_(rgb_conv.bias)
            self.encoder.downsample_layers[0][0] = rgbd_conv

        # 冻结前两个特征阶段；输入 stem 保持可训练，让深度通道参与学习。
        self.encoder.stages[0].requires_grad_(False)
        self.encoder.stages[1].requires_grad_(False)
        self.encoder.downsample_layers[1].requires_grad_(False)
        # 自定义四尺度 forward 不经过官方最终 norm；冻结它，避免 DDP 等待不存在的梯度。
        self.encoder.norm.requires_grad_(False)
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            self.encoder.embed_dims, self.out_channels, extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, images: torch.Tensor) -> OrderedDict:
        """参数：images 是已归一化的 B×C×H×W 图像，C 为 3 或 4。

        返回：FPN 特征字典，键为 0/1/2/3/pool，对应五种特征尺度。
        """
        features = OrderedDict()
        x = images
        for index, (downsample, stage) in enumerate(
            zip(self.encoder.downsample_layers, self.encoder.stages)
        ):
            x = stage(downsample(x))
            features[str(index)] = x
        return self.fpn(features)


def build_model(config: dict, pretrained: bool = False) -> MaskRCNN:
    """构建单类元器件实例分割模型，背景标签为 0，元器件标签为 1。

    参数：
        config：包含 backbone_repo、backbone_weights、use_depth、
            small_anchors、image_size 的扁平配置；相对路径以项目根目录为准。
        pretrained：新训练时加载官方 DINOv3 权重；恢复完整 checkpoint 时为 False。
    返回：
        MaskRCNN。训练输入图像与目标列表，返回损失字典；推理只输入图像列表，
        返回每张图的 boxes、labels、scores 和 masks。输入通道为 RGB 或 RGB-D。
    """
    backbone = DinoBackbone(config, pretrained)
    sizes = ((16,), (32,), (64,), (128,), (256,)) if config["small_anchors"] else (
        (32,), (64,), (128,), (256,), (512,),
    )
    ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    mean = [0.485, 0.456, 0.406] + ([0.0] if config["use_depth"] else [])
    std = [0.229, 0.224, 0.225] + ([1.0] if config["use_depth"] else [])
    return MaskRCNN(
        backbone, num_classes=2,
        min_size=config["image_size"], max_size=config["image_size"],
        image_mean=mean, image_std=std,
        rpn_anchor_generator=AnchorGenerator(sizes, ratios),
        box_score_thresh=0.001, box_detections_per_img=100,
    )
