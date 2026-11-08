"""Validate local SD1.5 Diffusers components before allocating GPU workers."""

import argparse
import hashlib
import json
from pathlib import Path


def load_options(variant):
    return {"variant": variant} if variant else {}


def validate_base(model, variant=""):
    model = Path(model)
    files = [model/name for name in ("unet/config.json", "vae/config.json", "text_encoder/config.json",
        "scheduler/scheduler_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json")]
    suffix = "."+variant if variant else ""
    for folder, prefix in (("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
        candidates = [model/folder/(prefix+suffix+".safetensors"),
                      model/folder/(("pytorch_model" if folder == "text_encoder" else prefix)+suffix+".bin")]
        weight = next((path for path in candidates if path.is_file()), None)
        if weight is None:
            raise FileNotFoundError("Missing Diffusers weights: "+str(candidates[0]))
        files.append(weight)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    unet = json.loads((model/"unet/config.json").read_text())
    if unet["in_channels"] != 4 or unet["out_channels"] != 4 or unet["cross_attention_dim"] != 768:
        raise ValueError("Need four-channel SD1.5 UNet with cross_attention_dim=768")
    configs = {str(path.relative_to(model)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.suffix == ".json"}
    return dict(in_channels=unet["in_channels"], cross_attention_dim=768, variant=variant,
                config_hashes=configs, weight_files=[str(path) for path in files if path.suffix in (".bin", ".safetensors")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", default="")
    args = parser.parse_args()
    result = validate_base(args.model, args.variant)
    print(json.dumps(result, indent=2))
