"""Mask2Former RGB-D 实例分割训练入口：DDP 多卡、完整续训、轮末验证。

用法：
    python train_m2f.py --config config_m2f.json [--output ...] [--num-gpus N]
    python train_m2f.py --config config_m2f.json --resume /path/to/last.pth

每轮结束在完整 val 上计算 COCO 指标（与旧管线同一协议），按 Mask AP_s 保存
最佳模型；last.pth 保存完整训练状态（模型+优化器+调度器）用于续训。
"""
import argparse
from datetime import timedelta
import json
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from m2f_rgbd.data import RGBDInstanceDataset, collate_targets
from m2f_rgbd.engine import collect_predictions, evaluate_predictions, train_one_epoch
from m2f_rgbd.model import build_mask2former
from m2f_rgbd.utils import (
    StepWarmupScheduler, distributed_val_subset, load_checkpoint, make_loader,
    project_path, save_checkpoint, setup_device, write_json,
)


def choose_workers(value, local_processes):
    """按 value（整数或 auto）及本机 GPU 进程数，返回每个进程的数据 worker 数。"""
    if value != "auto":
        return int(value)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocation.isdigit():
        cpus = min(cpus, int(allocation))
    return min(4, max(0, cpus // max(local_processes, 1) - 1))


def train_worker(local_rank, config, world_size=1, init_method=None):
    """在 local_rank 对应 GPU 上训练；config 为配置，world_size 为总进程数。"""
    rank = int(os.environ["RANK"]) if init_method == "env://" else local_rank
    distributed = world_size > 1
    requested = f"cuda:{local_rank}" if config["device"] != "cpu" and torch.cuda.is_available() else config["device"]
    device = setup_device(requested, config["seed"] + rank)
    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
    cpu_group = None
    try:
        if distributed:
            if device.type != "cuda" or not dist.is_nccl_available():
                raise RuntimeError("Multi-GPU training requires a Linux CUDA PyTorch installation with NCCL.")
            timeout = timedelta(minutes=60)
            dist.init_process_group("nccl", init_method=init_method, rank=rank,
                                    world_size=world_size, timeout=timeout)
            cpu_group = dist.new_group(backend="gloo", timeout=timeout)

        output = project_path(config["output_dir"])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)

        local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        workers = choose_workers(config["workers"], local_processes)
        root = project_path(config["data_root"])
        train_set = RGBDInstanceDataset(
            root, "train", mode=config["mode"], jitter=tuple(config["jitter"]),
            crop_size=config["crop_size"], small_crop_prob=config.get("small_crop_prob", 0.5),
            limit=config["limit_train"],
        )
        val_set = RGBDInstanceDataset(
            root, "val", mode=config["mode"], jitter=None, crop_size=None,
            limit=config["limit_val"],
        )
        if set(train_set.ids) & set(val_set.ids):
            raise ValueError("Train and validation image IDs overlap.")
        if len(train_set) < world_size:
            raise ValueError("Training images must be at least the number of GPUs; increase --limit-train.")

        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                     seed=config["seed"], shuffle=True, drop_last=True) if distributed else None
        train_loader = make_loader(train_set, config["batch_size"], workers,
                                   shuffle=True, sampler=sampler, drop_last=True,
                                   collate_fn=collate_targets)
        val_subset = distributed_val_subset(val_set, rank, world_size)
        val_loader = make_loader(val_subset, config["batch_size"], workers,
                                 collate_fn=collate_targets)
        ground_truth = val_set.build_coco() if rank == 0 else None

        model = build_mask2former(
            project_path(config["checkpoint_dir"]),
            num_channels=config["num_channels"],
            num_labels=config["num_labels"],
        ).to(device)
        init_path = project_path(config["init_from"]) if config.get("init_from") else None
        if init_path is not None and not config.get("resume"):
            model.load_state_dict(load_checkpoint(init_path)["model"])
            if rank == 0:
                print(f"Initialized weights from {init_path}; optimizer/scheduler start fresh", flush=True)
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=False) if distributed else model

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            betas=(0.9, 0.999), weight_decay=config["weight_decay"],
        )
        epoch_images = len(sampler) * world_size if distributed else len(train_set)
        total_iters = max(1, int(epoch_images // (config["batch_size"] * world_size * config["grad_accum"]))) * config["epochs"]
        scheduler = StepWarmupScheduler(
            optimizer, config["learning_rate"], total_iters,
            warmup_iters=config["warmup_iters"], milestones=(0.9, 0.95),
        )
        best_score = -float("inf")
        start_epoch = 0

        resume_path = project_path(config["resume"]) if config.get("resume") else None
        if resume_path is not None:
            state = load_checkpoint(resume_path)
            raw = model.module if distributed else model
            raw.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            best_score = state.get("best_score", -float("inf"))
            if rank == 0:
                print(f"Resumed from {resume_path} at epoch {start_epoch}", flush=True)

        if rank == 0:
            write_json(output / "config.json", config)
            write_json(output / "environment.json", {
                "torch": torch.__version__, "device": str(device), "world_size": world_size,
                "gpu_names": [torch.cuda.get_device_name(index) for index in range(min(local_processes, torch.cuda.device_count()))]
                             if device.type == "cuda" else [],
                "batch_size_per_gpu": config["batch_size"],
                "grad_accum": config["grad_accum"],
                "effective_batch_size": config["batch_size"] * world_size * config["grad_accum"],
                "workers_per_gpu": workers, "amp_dtype": config["amp_dtype"],
                "training_images": len(train_set), "training_images_per_epoch": epoch_images,
                "validation_images": len(val_set),
            })
            print(f"GPUs={world_size if device.type == 'cuda' else 0}; batch/GPU={config['batch_size']}; "
                  f"accum={config['grad_accum']}; effective batch={config['batch_size'] * world_size * config['grad_accum']}; "
                  f"workers/GPU={workers}; output={output}", flush=True)

        for epoch in range(start_epoch, config["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if rank == 0:
                print(f"Epoch {epoch + 1}/{config['epochs']} lr={scheduler.get_last_lr()[0]:.3e}", flush=True)
            started = time.perf_counter()
            losses = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch + 1,
                amp_dtype=config["amp_dtype"], grad_accum=config["grad_accum"],
                max_grad_norm=config["max_grad_norm"],
            )
            training_seconds = time.perf_counter() - started

            started = time.perf_counter()
            raw_model = model.module if distributed else model
            predictions = collect_predictions(
                raw_model, val_loader, device, score_threshold=config["score_threshold"],
                quiet=rank != 0, amp_dtype=config["amp_dtype"],
            )
            if distributed:
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(predictions, gathered, dst=0, group=cpu_group)
                if rank == 0:
                    predictions = [record for shard in gathered for record in shard]
                    del gathered
            if rank == 0:
                metrics = evaluate_predictions(ground_truth, predictions, val_set.ids)
                validation_seconds = time.perf_counter() - started
                score = metrics["segm"]["AP_s"]
                improved = score > best_score
                best_score = max(best_score, score)
                state = {"config": config, "epoch": epoch, "best_score": best_score,
                         "model": raw_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()}
                save_checkpoint(output / "last.pth", state)
                if improved:
                    save_checkpoint(output / "best.pth", {"config": config, "epoch": epoch,
                                                          "best_score": best_score,
                                                          "model": raw_model.state_dict()})
                row = {"epoch": epoch + 1, "losses": losses, "metrics": metrics,
                       "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                       "training_images_per_second": epoch_images / training_seconds}
                with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"Mask AP={metrics['segm']['AP']:.4f}; AP_s={score:.4f}; best={best_score:.4f}; "
                      f"train={training_seconds:.1f}s ({epoch_images / training_seconds:.2f} images/s); "
                      f"validation={validation_seconds:.1f}s", flush=True)
            del predictions
            if distributed:
                dist.barrier(group=cpu_group)
    finally:
        if dist.is_initialized():
            if cpu_group is not None:
                dist.destroy_process_group(cpu_group)
            dist.destroy_process_group()


def main():
    """读取配置与命令行，启动每 GPU 一个训练进程；兼容 torchrun，无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_m2f.json")
    parser.add_argument("--output")
    parser.add_argument("--num-gpus", type=int, help="Maximum GPUs to use; default from config (1)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Images per GPU, not the global batch size")
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--init", help="Load model weights only and start a fresh optimizer/scheduler")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    with project_path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    overrides = {"output_dir": args.output, "num_gpus": args.num_gpus, "epochs": args.epochs,
                 "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                 "workers": args.workers, "limit_train": args.limit_train,
                 "limit_val": args.limit_val, "resume": args.resume,
                 "init_from": args.init, "device": args.device}
    config.update({key: value for key, value in overrides.items() if value is not None})
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["grad_accum"] < 1:
        parser.error("epochs, batch_size and grad_accum must be positive")

    num_gpus = config.get("num_gpus", 1)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        parser.error("num_gpus must be a positive integer")
    output = project_path(config["output_dir"])
    if int(os.environ.get("RANK", 0)) == 0 and output.exists() and any(output.iterdir()) \
            and not config.get("resume"):
        parser.error(f"Output directory is not empty: {output}. Choose a new --output or pass --resume.")
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > num_gpus:
        parser.error("torchrun WORLD_SIZE exceeds num_gpus; use one process for single-GPU training")

    if "LOCAL_RANK" in os.environ:
        train_worker(int(os.environ["LOCAL_RANK"]), config,
                     int(os.environ["WORLD_SIZE"]), "env://")
        return
    gpu_count = min(num_gpus, torch.cuda.device_count()) if config["device"] != "cpu" else 0
    if gpu_count > 1:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(train_worker, args=(config, gpu_count, f"tcp://127.0.0.1:{port}"),
                 nprocs=gpu_count, join=True)
    else:
        train_worker(0, config)


if __name__ == "__main__":
    main()
