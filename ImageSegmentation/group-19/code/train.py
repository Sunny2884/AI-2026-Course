"""每次从预训练权重开始训练；按 num_gpus 限制使用可见 GPU，默认单卡。"""
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
from torch.utils.data import DistributedSampler, Subset

from rgbdseg.data import ComponentDataset
from rgbdseg.engine import collect_predictions, evaluate_predictions, train_one_epoch
from rgbdseg.model import build_model
from rgbdseg.utils import make_loader, project_path, save_checkpoint, setup_device, write_json


def choose_workers(value, local_processes):
    """按 value（整数或 auto）及本机 GPU 进程数，返回每个进程的数据 worker 数。"""
    if value != "auto":
        return int(value)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocation.isdigit():
        cpus = min(cpus, int(allocation))
    # 为每张卡的主进程保留一个 CPU；每卡最多两个预取进程，控制原图 mask 的内存。
    return min(2, max(0, cpus // max(local_processes, 1) - 1))


def train_worker(local_rank, config, world_size=1, init_method=None):
    """在 local_rank 对应 GPU 上训练；config 为配置，world_size 为总进程数，无返回值。"""
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
            # RLE/JSON 在 CPU 汇总，避免通过 NCCL 将大型 Python 对象搬入显存。
            cpu_group = dist.new_group(backend="gloo", timeout=timeout)

        output = project_path(config["output_dir"])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)

        local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        workers = choose_workers(config["workers"], local_processes)
        root = project_path(config["data_root"])
        train_set = ComponentDataset(root, "train", use_depth=config["use_depth"], augment=True,
                                     limit=config["train_limit"])
        val_set = ComponentDataset(root, "val", use_depth=config["use_depth"], limit=config["val_limit"])
        if set(train_set.ids) & set(val_set.ids):
            raise ValueError("Train and validation image IDs overlap.")
        train_scenes = {image["scene_id"] for image in train_set.images if "scene_id" in image}
        val_scenes = {image["scene_id"] for image in val_set.images if "scene_id" in image}
        if train_scenes & val_scenes:
            raise ValueError("Train and validation scenes overlap.")
        if len(train_set) < world_size:
            raise ValueError("Training images must be at least the number of GPUs; increase --limit-train.")

        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                     seed=config["seed"], shuffle=True, drop_last=True) if distributed else None
        train_loader = make_loader(train_set, config["batch_size"], workers,
                                   shuffle=True, sampler=sampler)
        # 验证按步长分片，既不补齐重复图像，也不丢弃尾部图像。
        val_subset = Subset(val_set, list(range(rank, len(val_set), world_size)))
        val_loader = make_loader(val_subset, 1, workers)
        ground_truth = val_set.build_coco() if rank == 0 else None

        raw_model = build_model(config, pretrained=config["pretrained"]).to(device)
        model = DistributedDataParallel(raw_model, device_ids=[local_rank],
                                        broadcast_buffers=False, gradient_as_bucket_view=True) if distributed else raw_model
        backbone_params, head_params = [], []
        for name, parameter in raw_model.named_parameters():
            if parameter.requires_grad:
                is_backbone = name.startswith("backbone.encoder.") and "downsample_layers.0." not in name
                (backbone_params if is_backbone else head_params).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config["learning_rate"] * 0.1},
            {"params": head_params, "lr": config["learning_rate"]},
        ], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"], gamma=0.1)
        amp = config["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best_score = -float("inf")
        epoch_images = len(sampler) * world_size if distributed else len(train_set)
        if rank == 0:
            write_json(output / "config.json", config)
            write_json(output / "environment.json", {
                "torch": torch.__version__, "device": str(device), "world_size": world_size,
                "gpu_names": [torch.cuda.get_device_name(index) for index in range(min(local_processes, torch.cuda.device_count()))]
                             if device.type == "cuda" else [],
                "batch_size_per_gpu": config["batch_size"],
                "global_batch_size": config["batch_size"] * world_size,
                "workers_per_gpu": workers, "amp": amp,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "training_images": len(train_set), "training_images_per_epoch": epoch_images,
                "validation_images": len(val_set),
            })
            print(f"GPUs={world_size if device.type == 'cuda' else 0}; batch/GPU={config['batch_size']}; "
                  f"global batch={config['batch_size'] * world_size}; workers/GPU={workers}; output={output}", flush=True)

        for epoch in range(config["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if rank == 0:
                print(f"Epoch {epoch + 1}/{config['epochs']}", flush=True)
            started = time.perf_counter()
            losses = train_one_epoch(model, train_loader, optimizer, scaler, device, amp)
            training_seconds = time.perf_counter() - started
            scheduler.step()

            started = time.perf_counter()
            # 各 rank 的验证批数可以不同，因此使用未包装模型，避免 DDP forward 同步。
            predictions = collect_predictions(raw_model, val_loader, device, quiet=rank != 0, amp=amp)
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
                # 只保存推理必需的模型及配置；不保存优化器，也不支持续训。
                state = {"config": config, "epoch": epoch, "best_score": best_score,
                         "model": raw_model.state_dict()}
                save_checkpoint(output / "last.pth", state)
                if improved:
                    save_checkpoint(output / "best.pth", state)
                row = {"epoch": epoch + 1, "losses": losses, "metrics": metrics,
                       "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                       "training_images_per_second": epoch_images / training_seconds}
                with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"Mask AP_s={score:.4f}; best={best_score:.4f}; "
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
    """读取配置与命令行，自动启动每 GPU 一个训练进程；兼容 torchrun，无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-gpus", type=int, help="Maximum GPUs to use; default from config (1)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, help="Images per GPU, not the global batch size")
    parser.add_argument("--workers", type=int, help="Data workers per GPU; default automatically follows the CPU allocation")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--rgb-only", action="store_true")
    parser.add_argument("--default-anchors", action="store_true")
    args = parser.parse_args()
    with project_path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    overrides = {"data_root": args.data_root, "output_dir": args.output, "device": args.device,
                 "num_gpus": args.num_gpus, "epochs": args.epochs, "image_size": args.image_size, "batch_size": args.batch_size,
                 "workers": args.workers, "train_limit": args.limit_train, "val_limit": args.limit_val}
    config.update({key: value for key, value in overrides.items() if value is not None})
    if config["epochs"] < 1 or config["batch_size"] < 1:
        parser.error("epochs and batch_size must be positive")
    if args.rgb_only:
        config["use_depth"] = False
    if args.default_anchors:
        config["small_anchors"] = False
    if args.no_pretrained:
        config["pretrained"] = False

    num_gpus = config.get("num_gpus", 1)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        parser.error("num_gpus must be a positive integer")
    output = project_path(config["output_dir"])
    if int(os.environ.get("RANK", 0)) == 0 and output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. Choose a new --output.")
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > num_gpus:
        parser.error("torchrun WORLD_SIZE exceeds num_gpus; use one process for single-GPU training")

    if "LOCAL_RANK" in os.environ:
        # torchrun 已经启动进程，沿用它的 GPU/节点分配，不再次 spawn。
        train_worker(int(os.environ["LOCAL_RANK"]), config,
                     int(os.environ["WORLD_SIZE"]), "env://")
        return
    gpu_count = min(num_gpus, torch.cuda.device_count()) if config["device"] != "cpu" else 0
    if gpu_count > 1:
        # 选择当前空闲端口，减少同一节点多个训练作业的端口冲突。
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(train_worker, args=(config, gpu_count, f"tcp://127.0.0.1:{port}"),
                 nprocs=gpu_count, join=True)
    else:
        train_worker(0, config)


if __name__ == "__main__":
    main()
