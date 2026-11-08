"""Train CLIP-only IP-Adapter and LoRA against selected anonymous targets."""

import argparse
import csv
import json
import secrets
import hashlib
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import FEATURE_SCHEMA, FEATURE_SHAPES, MultiFeatureAdapter, ConditionalDenoiser, validate_features
from safetensors.torch import load_file
from safetensors import safe_open
from checkpointing import install_compact_checkpoint_hooks
from run_id import new_run_id
from monitoring import TrainingMonitor
from initialization import initialize, vae_signature
from cache_io import read_features, validate_cached_row
from base_model import validate_base, load_options
from ip_adapter import GENERATOR_SCHEMA, IP_TOKENS, read_ip_weights, install_ip_attention, image_attention_state, empty_context


class PairsDataset(Dataset):
    def __init__(self, manifest, data_root, train_csv, resolution):
        self.resolution = resolution
        with train_csv.open(newline="", encoding="utf-8-sig") as handle:
            allowed = {(data_root/r["image_path"]).resolve() for r in csv.DictReader(handle)}
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("Empty training manifest")
        seen = set()
        for row in self.rows:
            source = Path(row["source_path"]).resolve()
            target = Path(row.get("target_path", ""))
            if source not in allowed or source in seen:
                raise ValueError("Non-training or duplicate source in manifest: " + str(source))
            seen.add(source)
            if not target.is_absolute() or not target.is_file() or target.resolve() == source:
                raise ValueError("A selected anonymous teacher target is required")
            validate_cached_row(row)
        fingerprints = {row["feature_fingerprint"] for row in self.rows}
        if len(fingerprints) != 1:
            raise ValueError("Mixed feature cache configurations")
        self.fingerprint = next(iter(fingerprints))
        if not self.rows:
            raise ValueError("Empty training manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Source RGB is handled offline by frozen encoders, never by the denoiser.
        with Image.open(row["target_path"]) as source:
            rgb = np.array(source.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS))
        target = torch.from_numpy(rgb).permute(2, 0, 1).float()/127.5-1
        return {"target": target, "condition": read_features(row)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "pairs", "data-root", "train-csv", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="bf16")
    parser.add_argument("--resume", type=Path, help="Complete epoch checkpoint with accelerate_state")
    parser.add_argument("--run-id")
    parser.add_argument("--ip-adapter-weight", type=Path, required=True)
    parser.add_argument("--ip-scale", type=float, default=1.)
    parser.add_argument("--model-variant", default="")
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--histogram-every", type=int, default=100)
    parser.add_argument("--image-every", type=int, default=100)
    parser.add_argument("--training-vae",type=Path,required=True)
    parser.add_argument("--init-checkpoint",type=Path)
    args = parser.parse_args()
    if args.resume and args.init_checkpoint: raise ValueError("RESUME and INIT_CHECKPOINT are mutually exclusive")
    from torch.utils.tensorboard import SummaryWriter  # Fail on every rank if unavailable.
    from accelerate import Accelerator
    from accelerate.utils import set_seed, broadcast_object_list, DistributedDataParallelKwargs
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("Train through a GPU job")
    if min(args.batch_size, args.accumulation, args.epochs) < 1 or args.resolution % 8:
        raise ValueError("Invalid batch, accumulation, epoch or resolution")
    if args.resolution < 64 or args.max_steps < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid resolution, step budget or warmup")
    if args.log_every < 1 or min(args.histogram_every, args.image_every) < 0:
        raise ValueError("Invalid monitoring intervals")
    if not math.isfinite(args.ip_scale) or args.ip_scale <= 0:
        raise ValueError("IP scale must be finite and positive")
    pretrained_ip = read_ip_weights(args.ip_adapter_weight)
    base_signature = validate_base(args.model, args.model_variant)
    if not 0 < args.condition_dropout < 1: raise ValueError("Condition dropout must lie in (0,1)")
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation, mixed_precision=args.mixed_precision,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    previous = None
    if args.resume:
        if not (args.resume/"complete.json").is_file():
            raise ValueError("Resume requires a complete epoch checkpoint")
        previous = json.loads((args.resume/"config.json").read_text())
        args.seed = previous["seed"]
        args.run_id = previous.get("run_id") or args.run_id
    if args.run_id is None:
        run_ids = [new_run_id() if accelerator.is_main_process else None]
        args.run_id = broadcast_object_list(run_ids)[0]
    accelerator.print("Run ID: {}; output: {}".format(args.run_id, args.output))
    if args.seed is None:
        seed_values = [secrets.randbits(32) if accelerator.is_main_process else None]
        args.seed = broadcast_object_list(seed_values)[0]
    accelerator.print("Training seed: {}".format(args.seed))
    set_seed(args.seed)
    output_status = [None]
    if accelerator.is_main_process:
        try:
            args.output.mkdir(parents=True, exist_ok=bool(args.resume))
        except OSError as error:
            output_status[0] = str(error)
    if broadcast_object_list(output_status)[0]:
        raise ValueError(output_status[0])
    accelerator.wait_for_everyone()
    data = PairsDataset(args.pairs, args.data_root, args.train_csv, args.resolution)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    unet = UNet2DConditionModel.from_pretrained(str(args.model), subfolder="unet", local_files_only=True,
                                              **load_options(args.model_variant))
    unet.requires_grad_(False)
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout, init_lora_weights="gaussian",
                             target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora_config)
    install_ip_attention(unet, pretrained_ip["ip_adapter"], scale=args.ip_scale)
    unet.enable_gradient_checkpointing()
    adapter = MultiFeatureAdapter(cross_attention_dim=unet.config.cross_attention_dim)
    adapter.resampler.load_state_dict(pretrained_ip["image_proj"], strict=True)
    del pretrained_ip
    model = ConditionalDenoiser(unet, adapter, empty_context(args.model, args.model_variant))
    cache_metadata=json.loads((args.pairs.parent/"config.json").read_text())
    warm_start=None
    if args.init_checkpoint:
        warm_start=initialize(model,args.init_checkpoint,{**vars(args),"feature_schema":FEATURE_SCHEMA,
            "generator_schema":GENERATOR_SCHEMA,"encoder_provenance":cache_metadata["encoders"],
            "preprocessing":cache_metadata["preprocessing"],"base_signature":base_signature,"model":str(args.model)})
    vae = AutoencoderKL.from_pretrained(str(args.training_vae), local_files_only=True)
    if vae.config.latent_channels!=4 or not np.isclose(vae.config.scaling_factor,.18215):raise ValueError("Invalid training VAE")
    vae.requires_grad_(False)
    vae.eval().to(accelerator.device, dtype=torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(str(args.model), subfolder="scheduler", local_files_only=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in unet.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": args.adapter_learning_rate},
    ], weight_decay=args.weight_decay)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    updates_per_epoch = math.ceil(len(loader)/args.accumulation)
    total_steps = args.max_steps or updates_per_epoch*args.epochs
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer,
                                 num_warmup_steps=int(total_steps*args.warmup_ratio),
                                 num_training_steps=total_steps)
    accelerator.register_for_checkpointing(lr_scheduler)
    install_compact_checkpoint_hooks(accelerator)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cache_config = json.loads((args.pairs.parent/"config.json").read_text())
    cache_complete = json.loads((args.pairs.parent/"complete.json").read_text())
    if cache_config["fingerprint"] != data.fingerprint or cache_config["schema"] != FEATURE_SCHEMA:
        raise ValueError("Feature cache config/manifest mismatch")
    if cache_complete["manifest_sha256"] != hashlib.sha256(args.pairs.read_bytes()).hexdigest():
        raise ValueError("Feature manifest changed after cache completion")
    config.update(feature_schema=FEATURE_SCHEMA, feature_shapes=FEATURE_SHAPES,
                  feature_fingerprint=data.fingerprint, encoder_provenance=cache_config["encoders"],
                  preprocessing=cache_config["preprocessing"],
                  cross_attention_dim=unet.config.cross_attention_dim, total_condition_tokens=IP_TOKENS,
                  clip_input_tokens=257, generator_schema=GENERATOR_SCHEMA,
                  base_signature=base_signature, training_vae_signature=vae_signature(args.training_vae),
                  initialization=warm_start,
                  ip_adapter_sha256=hashlib.sha256(args.ip_adapter_weight.read_bytes()).hexdigest(),
                  global_batch=args.batch_size*args.accumulation*accelerator.num_processes,
                  world_size=accelerator.num_processes, updates_per_epoch=updates_per_epoch,
                  checkpoint_format="trainable_only_v1",
                  total_steps=total_steps, pairs_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                  train_csv_sha256=hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  num_pairs=len(data), loss="mean squared diffusion noise or velocity prediction")
    if previous:
        for key in ("pairs_sha256", "train_csv_sha256", "global_batch", "world_size", "resolution",
                    "lora_rank", "lora_alpha", "lora_dropout", "feature_schema", "feature_fingerprint", "total_steps", "model",
                    "learning_rate", "adapter_learning_rate", "mixed_precision", "generator_schema",
                    "ip_adapter_sha256", "ip_scale", "model_variant", "condition_dropout", "base_signature", "training_vae_signature"):
            if config[key] != previous[key]:
                raise ValueError("Resume configuration mismatch: " + key)
        accelerator.load_state(str(args.resume/"accelerate_state"))
    if accelerator.is_main_process:
        (args.output/"config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
        (args.output/"training_report.md").write_text("# MSE VAE training v5\n\nTrainable Resampler, IP image attention and UNet LoRA. Null condition: zero raw CLIP tokens before Resampler; 10% dropout by default. No auxiliary features.\n")
    optimizer.zero_grad(set_to_none=True)
    global_step = previous["step"] if previous else 0
    start_epoch = previous["epoch"] if previous else 0
    monitor = TrainingMonitor(accelerator, args.output, {**config, "resume_step": global_step+1 if previous else None},
                              args.log_every, args.histogram_every, args.image_every)
    for epoch in range(start_epoch, args.epochs):
        monitor.start_epoch()
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                    latents = vae.encode(batch["target"].float()).latent_dist.sample()*vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                           (latents.shape[0],), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                with accelerator.autocast():
                    keep = (torch.rand(len(noisy),device=noisy.device) >= args.condition_dropout)[:,None,None]
                    prediction = model(noisy, timesteps, {"clip": batch["condition"]["clip"]*keep})
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError("Unsupported diffusion prediction type")
                per_sample_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
                loss = per_sample_loss.mean()
                if not torch.isfinite(loss):
                    accelerator.set_trigger()
                if accelerator.check_trigger():
                    raise FloatingPointError("Non-finite training loss on at least one rank")
                monitor.observe(per_sample_loss, timesteps, noise_scheduler.config.num_train_timesteps,
                                prediction, target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if not torch.isfinite(grad_norm):
                        accelerator.set_trigger()
                    if accelerator.check_trigger():
                        raise FloatingPointError("Non-finite gradient norm on at least one rank")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                    global_step += 1
                    monitor.update(global_step, epoch+1, lr_scheduler.get_last_lr(), grad_norm,
                                   model, batch["condition"], timesteps)
                    monitor.images(global_step, batch, noisy, prediction, timesteps, vae, noise_scheduler)
            if global_step >= total_steps:
                break
        monitor.epoch_end(epoch+1, global_step)
        accelerator.wait_for_everyone()
        checkpoint = args.output/("epoch_{:03d}".format(epoch+1))
        checkpoint_started = time.perf_counter()
        accelerator.print("Epoch {} finished, step {}; checkpoint save starting: {}".format(epoch+1,global_step,checkpoint), flush=True)
        if accelerator.is_main_process:
            checkpoint.mkdir(exist_ok=False)
        accelerator.wait_for_everyone()
        accelerator.save_state(str(checkpoint/"accelerate_state"))
        accelerator.wait_for_everyone()
        accelerator.print("Optimizer, scheduler, RNG and trainable weights saved; exporting LoRA/adapter", flush=True)
        if accelerator.is_main_process:
            raw = accelerator.unwrap_model(model)
            lora = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(raw.unet).items()}
            save_file(lora, str(checkpoint/"unet_lora.safetensors"))
            save_file(image_attention_state(raw.unet), str(checkpoint/"unet_ip_adapter.safetensors"))
            save_file({"null_context": raw.null_context.detach().cpu().contiguous()},
                      str(checkpoint/"base_context.safetensors"))
            save_file({k: v.detach().cpu().contiguous() for k, v in raw.adapter.state_dict().items()},
                      str(checkpoint/"condition_adapter.safetensors"))
            (checkpoint/"config.json").write_text(json.dumps({**config, "epoch": epoch+1, "step": global_step}, indent=2)+"\n")
            (checkpoint/"complete.json").write_text(json.dumps(dict(epoch=epoch+1,step=global_step))+"\n")
        accelerator.wait_for_everyone()
        accelerator.print("Checkpoint complete in {:.1f}s: {}".format(time.perf_counter()-checkpoint_started,checkpoint), flush=True)
        if monitor.writer:
            monitor.writer.add_scalar("performance/checkpoint_seconds", time.perf_counter()-checkpoint_started, global_step)
        if global_step >= total_steps:
            break
    monitor.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
