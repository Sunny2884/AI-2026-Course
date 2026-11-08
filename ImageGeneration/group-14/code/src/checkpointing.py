"""Compact Accelerate checkpoints for frozen-backbone LoRA training."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def trainable_state(model):
    state = {name: parameter.detach().cpu().contiguous()
             for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not state:
        raise ValueError("No trainable parameters to checkpoint")
    return state


def install_compact_checkpoint_hooks(accelerator):
    def save_hook(models, weights, output_dir):
        if len(models) != 1:
            raise ValueError("Compact checkpoint expects exactly one conditional denoiser")
        if accelerator.is_main_process:
            raw = accelerator.unwrap_model(models[0])
            state = trainable_state(raw)
            size = sum(t.numel()*t.element_size() for t in state.values())
            print("Checkpoint: writing {} trainable tensors ({:.1f} MiB); frozen UNet excluded".format(
                len(state), size/1024**2), flush=True)
            save_file(state, str(Path(output_dir)/"trainable.safetensors"))
        # Accelerate has already assembled the model state dictionaries. Drop
        # them here so it does not serialize the frozen multi-GB backbone.
        weights.clear()

    def load_hook(models, input_dir):
        path = Path(input_dir)/"trainable.safetensors"
        if not path.exists():
            # Completed legacy checkpoints still use Accelerate's full loader.
            return
        if len(models) != 1:
            raise ValueError("Compact checkpoint expects exactly one conditional denoiser")
        raw = accelerator.unwrap_model(models[0])
        state = load_file(str(path), device="cpu")
        parameters = {name: p for name, p in raw.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Trainable checkpoint parameter names do not match the model")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if state[name].shape != parameter.shape:
                    raise ValueError("Checkpoint shape mismatch: " + name)
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
