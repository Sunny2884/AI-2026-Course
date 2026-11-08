"""Weights-only warm-start, explicitly independent of Accelerate resume."""
import hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from ip_adapter import restore_image_attention
from lora_loading import load_lora

def vae_signature(path):
    path=Path(path)
    weight=next((path/n for n in ('diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin') if (path/n).is_file()),None)
    if weight is None:raise FileNotFoundError('Missing standalone VAE weights: '+str(path))
    h=hashlib.sha256()
    with weight.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return dict(path=str(path.resolve()),weights_sha256=h.hexdigest(),config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest())

def initialize(model,checkpoint,settings):
    checkpoint=Path(checkpoint)
    if not (checkpoint/'complete.json').is_file():raise ValueError('Warm-start requires a complete checkpoint')
    old=json.loads((checkpoint/'config.json').read_text())
    for k in ('model','model_variant','lora_rank','lora_alpha','lora_dropout','feature_schema','generator_schema','encoder_provenance','preprocessing','base_signature'):
        if old[k]!=settings[k]:raise ValueError('Warm-start configuration mismatch: '+k)
    files={name:checkpoint/name for name in ('unet_lora.safetensors','unet_ip_adapter.safetensors','condition_adapter.safetensors','base_context.safetensors')}
    states={name:load_file(str(path)) for name,path in files.items()}
    load_lora(model.unet,states['unet_lora.safetensors'])
    restore_image_attention(model.unet,states['unet_ip_adapter.safetensors'])
    model.adapter.load_state_dict(states['condition_adapter.safetensors'],strict=True)
    context=states['base_context.safetensors']['null_context']
    if context.shape!=model.null_context.shape or not torch.isfinite(context).all():raise ValueError('Invalid base context')
    model.null_context.copy_(context)
    return dict(checkpoint=str(checkpoint),source_epoch=old['epoch'],source_step=old['step'],
        sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
        optimizer_reset=True,scheduler_reset=True,step_reset=True)
