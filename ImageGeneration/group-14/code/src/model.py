"""CLIP-only IP-Adapter Plus, with one consistently trained null visual condition."""
import torch
from torch import nn
from ip_adapter import Resampler
FEATURE_SCHEMA = "clip_only_cfg_v4"
FEATURE_SHAPES = {"clip": (257,1280)}

def validate_features(features,batched=False):
    if set(features)!= {"clip"}:
        raise ValueError("Only CLIP features accepted")
    x=features["clip"]
    if x.ndim!=2+int(batched) or tuple(x.shape[-2:])!=(257,1280) or not torch.isfinite(x).all():
        raise ValueError("Invalid CLIP hidden states")

class MultiFeatureAdapter(nn.Module):
    def __init__(self,cross_attention_dim=768):
        super().__init__()
        self.resampler=Resampler(dim=cross_attention_dim,output_dim=cross_attention_dim,
            heads=12 if cross_attention_dim==768 else 4,dim_head=64 if cross_attention_dim==768 else cross_attention_dim//4)
    def forward(self,features):
        return self.resampler(features["clip"].float())

class ConditionalDenoiser(nn.Module):
    def __init__(self,unet,adapter,null_context=None):
        super().__init__()
        if unet.config.in_channels!=4 or unet.config.out_channels!=4:
            raise ValueError("Four-channel UNet required")
        self.unet,self.adapter=unet,adapter
        if null_context is None:
            null_context=torch.zeros(1,77,unet.config.cross_attention_dim)
        self.register_buffer("null_context",null_context,persistent=False)
    def forward(self,noisy,timestep,features):
        tokens=self.adapter(features)
        base=self.null_context.expand(len(noisy),-1,-1).to(tokens.dtype)
        return self.unet(noisy,timestep,encoder_hidden_states=(base,[tokens])).sample

def guided_prediction(model,noisy,timestep,features,scale):
    if scale==1:
        return model(noisy,timestep,features)
    clip=features["clip"]
    uncond,cond=model(torch.cat((noisy,noisy)),timestep,
        {"clip":torch.cat((torch.zeros_like(clip),clip))}).chunk(2)
    return uncond+scale*(cond-uncond)
