"""Strict loading of this project's non-sharded, default-adapter LoRA exports."""
import torch

def load_lora(unet,state):
    parameters={}
    for name,p in unet.named_parameters():
        if '.lora_A.default.' in name or '.lora_B.default.' in name:
            key=name.replace('.lora_A.default.','.lora_A.').replace('.lora_B.default.','.lora_B.')
            if key in parameters:raise ValueError('Duplicate LoRA export key: '+key)
            parameters[key]=p
    if not parameters or set(state)!=set(parameters):
        raise ValueError('LoRA key mismatch: missing={}, unexpected={}'.format(
            sorted(set(parameters)-set(state))[:5],sorted(set(state)-set(parameters))[:5]))
    # Validate the entire export before touching model weights.
    for key,p in parameters.items():
        value=state[key]
        if value.shape!=p.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid LoRA tensor: '+key)
    with torch.no_grad():
        for key,p in parameters.items():p.copy_(state[key].to(device=p.device,dtype=p.dtype))
    return len(parameters)
