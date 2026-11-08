"""CLIP-conditioned anonymous image generation with CFG."""

import argparse
import hashlib
import json
import secrets
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FEATURE_SCHEMA, MultiFeatureAdapter, ConditionalDenoiser, validate_features, guided_prediction
from run_id import new_run_id
from ip_adapter import GENERATOR_SCHEMA, install_ip_attention, restore_image_attention
from cache_io import read_features
from base_model import load_options, validate_base
from lora_loading import load_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "conditions", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--guidance-scale",type=float,default=3.)
    parser.add_argument("--vae", type=Path, default=None,
                        help="Standalone inference decoder; defaults to official MSE VAE")
    parser.add_argument("--sampler", choices=("ddim", "dpmpp_2m_karras"), default="ddim")
    parser.add_argument("--ip-scale", type=float)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.guidance_scale) or args.guidance_scale < 1: raise ValueError("Invalid CFG scale")
    if args.ip_scale is not None and (not np.isfinite(args.ip_scale) or args.ip_scale <= 0):
        raise ValueError("Invalid IP scale")
    if args.batch_size < 1 or args.steps < 1 or args.limit < 0:
        raise ValueError("Invalid batch size, diffusion steps or limit")
    if args.run_id is None:
        args.run_id = new_run_id()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    print("Inference seed: {}".format(args.seed), flush=True)
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
    from peft import LoraConfig
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("Run student generation on a GPU worker")
    config = json.loads((args.checkpoint/"config.json").read_text())
    if config["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("v4 checkpoint required")
    if config["generator_schema"] != GENERATOR_SCHEMA:
        raise ValueError("CLIP-only CFG v4 checkpoint required")
    if Path(config["model"]).resolve() != args.model.resolve():
        raise ValueError("Inference base model must match training")
    if not (args.checkpoint/"complete.json").is_file():
        raise ValueError("Incomplete checkpoint")
    feature_config = json.loads((args.conditions.parent/"config.json").read_text())
    if feature_config["encoders"] != config["encoder_provenance"] or feature_config["preprocessing"] != config["preprocessing"]:
        raise ValueError("Inference feature encoders/preprocessing differ from training")
    if validate_base(args.model, config["model_variant"]) != config["base_signature"]:
        raise ValueError("Base model config changed since training")
    unet = UNet2DConditionModel.from_pretrained(str(args.model), subfolder="unet", local_files_only=True,
                                              **load_options(config["model_variant"]))
    unet.add_adapter(LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                                lora_dropout=config["lora_dropout"],
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    install_ip_attention(unet, scale=args.ip_scale if args.ip_scale is not None else config["ip_scale"])
    restore_image_attention(unet, load_file(str(args.checkpoint/"unet_ip_adapter.safetensors")))
    load_lora(unet, load_file(str(args.checkpoint/"unet_lora.safetensors")))
    adapter = MultiFeatureAdapter(config["cross_attention_dim"])
    adapter.load_state_dict(load_file(str(args.checkpoint/"condition_adapter.safetensors")), strict=True)
    context = load_file(str(args.checkpoint/"base_context.safetensors"))["null_context"]
    if context.shape != (1, 77, config["cross_attention_dim"]) or not torch.isfinite(context).all():
        raise ValueError("Invalid fixed base context")
    model = ConditionalDenoiser(unet, adapter, context).to("cuda").eval()
    vae = AutoencoderKL.from_pretrained(str(args.model), subfolder="vae", local_files_only=True,
                                       **load_options(config["model_variant"])).to("cuda").eval()
    training_scaling_factor = vae.config.scaling_factor
    if args.vae:
        del vae
        vae = AutoencoderKL.from_pretrained(str(args.vae), local_files_only=True).to("cuda").eval()
        if vae.config.latent_channels != 4 or not np.isclose(vae.config.scaling_factor, training_scaling_factor):
            raise ValueError("Replacement decoder must use the same latent scale/channels as training")
    scheduler = (DDIMScheduler.from_pretrained(str(args.model), subfolder="scheduler", local_files_only=True)
        if args.sampler == "ddim" else DPMSolverMultistepScheduler.from_pretrained(str(args.model),
            subfolder="scheduler", local_files_only=True, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True))
    rows = [json.loads(line) for line in args.conditions.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("Empty conditions or invalid image IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(scheduler_config=dict(scheduler.config), training_scaling_factor=training_scaling_factor)
    (args.output/"config.json").write_text(json.dumps(run_config, indent=2)+"\n")
    resolution = config["resolution"]
    scale = 2**(len(vae.config.block_out_channels)-1)
    with torch.inference_mode(), (args.output/"manifest.jsonl").open("w") as manifest:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            seeds = [int.from_bytes(hashlib.sha256((str(args.seed)+":"+row["id"]).encode()).digest()[:8], "big") % (2**63)
                     for row in batch]
            cached = []
            for row in batch:
                if row["feature_fingerprint"] != feature_config["fingerprint"]:
                    raise ValueError("Cached CLIP config mismatch")
                cached.append(read_features(row))
            condition = {name: torch.stack([value[name] for value in cached]).to("cuda", dtype=torch.float32)
                         for name in cached[0]}
            scheduler.set_timesteps(args.steps, device="cuda")
            latent = torch.cat([torch.randn((1, 4, resolution//scale, resolution//scale),
                                generator=torch.Generator("cuda").manual_seed(seed), device="cuda")
                                for seed in seeds]) * scheduler.init_noise_sigma
            for timestep in scheduler.timesteps:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = guided_prediction(model, scheduler.scale_model_input(latent, timestep), timestep, condition, args.guidance_scale)
                options = {"eta": 0} if args.sampler == "ddim" else {}
                latent = scheduler.step(prediction.float(), timestep, latent, **options).prev_sample
            if args.save_latents:
                from safetensors.torch import save_file
                directory = args.output/"latents"
                directory.mkdir(exist_ok=True)
                for row, value in zip(batch, latent):
                    save_file({"latent": value.detach().cpu().contiguous()}, str(directory/(row["id"]+".safetensors")))
            decoded = vae.decode(latent/vae.config.scaling_factor).sample
            rgb = ((decoded.float().clamp(-1, 1)+1)*127.5).permute(0, 2, 3, 1).cpu().numpy().round().astype(np.uint8)
            for row, seed, image in zip(batch, seeds, rgb):
                Image.fromarray(image).save(args.output/(row["id"]+".png"))
                manifest.write(json.dumps({"id": row["id"], "seed": seed})+"\n")
            manifest.flush()
            print("student inference {}/{}".format(start+len(batch), len(rows)), flush=True)


if __name__ == "__main__":
    main()
