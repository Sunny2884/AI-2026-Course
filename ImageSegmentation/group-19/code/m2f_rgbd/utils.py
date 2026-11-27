"""设备、随机种子、学习率调度、checkpoint 读写和 DataLoader 工具。"""
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

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


def make_loader(dataset, batch_size=1, workers=0, shuffle=False, sampler=None,
                drop_last=False, pin_memory=True, collate_fn=None):
    """创建 DataLoader；多进程读取保留 worker。collate_fn 缺省为默认堆叠。"""
    options = {"persistent_workers": workers > 0}
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, worker_init_fn=seed_worker,
                      pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn,
                      **options)


class StepWarmupScheduler:
    """线性 warmup + 两次 step 衰减（Mask2Former 官方 LSJ 配方）。

    base_lr 在 warmup_iters 内从 1e-6 线性升到 base_lr；之后在 total_iters 的
    90% 与 95% 处各衰减 0.1 倍。每次 optimizer.step 后调用一次 step()。
    """

    def __init__(self, optimizer, base_lr, total_iters, warmup_iters=250, milestones=(0.9, 0.95)):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.milestones = milestones
        self.updates = 0

    def _apply(self):
        if self.updates <= self.warmup_iters:
            ratio = self.updates / max(self.warmup_iters, 1)
            scale = 1e-6 / self.base_lr + (1 - 1e-6 / self.base_lr) * ratio
        else:
            scale = 1.0
            for milestone in self.milestones:
                if self.updates >= milestone * self.total_iters:
                    scale *= 0.1
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale

    def step(self):
        self.updates += 1
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {"updates": self.updates, "base_lr": self.base_lr,
                "total_iters": self.total_iters, "warmup_iters": self.warmup_iters,
                "milestones": self.milestones}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._apply()


def save_checkpoint(path, state):
    """将训练状态 state 原子保存到 path，避免中断留下半个权重文件；无返回值。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    """从 path 读取本管线生成的可信 checkpoint 并返回字典。"""
    return torch.load(project_path(path), map_location="cpu", weights_only=False)


def write_json(path, value, indent=2):
    """将 value 保存为 UTF-8 JSON 到 path，自动创建父目录；无返回值。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=indent)


def distributed_val_subset(dataset, rank, world_size):
    """按步长分片验证集，既不补齐重复图像，也不丢弃尾部图像。"""
    return Subset(dataset, list(range(rank, len(dataset), world_size)))
