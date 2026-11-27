"""路径、随机种子、DataLoader 和文件读写，供三个入口脚本共用。"""
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """将路径 value 转为 Path；相对路径以项目根目录为起点，返回绝对路径。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def setup_device(requested="auto", seed=42):
    """设置 seed 随机种子，按 requested（auto/cpu/cuda）选择并返回 torch.device。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA PyTorch wheel or use --device cpu.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return device


def seed_worker(worker_id):
    """用 PyTorch 分配的种子初始化编号 worker_id 的数据进程；无返回值。"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size=1, workers=0, shuffle=False, sampler=None):
    """按 batch_size/workers 创建 dataset 的 DataLoader，返回保留变长标签的加载器。

    参数 shuffle 控制单进程随机顺序；传入 DDP sampler 时由 sampler 决定分片和顺序。
    多进程读取保留 worker，并让每个 worker 最多预取一批，限制实例掩码的内存占用。
    """
    options = {"persistent_workers": workers > 0}
    if workers > 0:
        options["prefetch_factor"] = 1
        options["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, collate_fn=collate_fn,
                      worker_init_fn=seed_worker, pin_memory=torch.cuda.is_available(), **options)


def write_json(path, value, indent=2):
    """将 value 保存为 UTF-8 JSON 到 path，自动创建父目录；无返回值。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=indent)


def save_checkpoint(path, state):
    """将训练状态 state 原子保存到 path，避免中断留下半个权重文件；无返回值。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    """从 path 读取本项目生成的可信 checkpoint 并返回字典；不接受未知来源的文件。"""
    return torch.load(project_path(path), map_location="cpu", weights_only=False)


def load_model(checkpoint_path, requested_device="auto", backbone_repo=None):
    """读取整模型权重，返回 (model, config, device)；可覆盖 DINOv3 源码路径。"""
    from .model import build_model

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"].copy()
    if backbone_repo is not None:
        config["backbone_repo"] = backbone_repo
    device = setup_device(requested_device, config["seed"])
    # 完整 checkpoint 已含 DINOv3 权重；此处无需原始预训练文件，也不联网。
    model = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, device
