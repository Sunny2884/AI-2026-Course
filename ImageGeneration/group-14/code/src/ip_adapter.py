"""IP-Adapter Plus-compatible Resampler and pretrained image attention loading.

Resampler layout follows Tencent AI Lab IP-Adapter (Apache-2.0):
https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
Uses PyTorch SDPA without an additional einops dependency.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


GENERATOR_SCHEMA = "clip_only_cfg_v4_1"
IP_TOKENS = 16
IP_INPUT_DIM = 1280


class PerceiverAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, heads*dim_head, bias=False)
        self.to_kv = nn.Linear(dim, 2*heads*dim_head, bias=False)
        self.to_out = nn.Linear(heads*dim_head, dim, bias=False)

    def forward(self, features, latents):
        features, latents = self.norm1(features), self.norm2(latents)
        query = self.to_q(latents)
        key, value = self.to_kv(torch.cat((features, latents), dim=1)).chunk(2, dim=-1)
        def split(tensor):
            return tensor.reshape(tensor.shape[0], -1, self.heads, self.dim_head).transpose(1, 2)
        output = F.scaled_dot_product_attention(split(query), split(key), split(value), dropout_p=0.)
        return self.to_out(output.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class Resampler(nn.Module):
    def __init__(self, dim=768, depth=4, heads=12, dim_head=64,
                 num_queries=IP_TOKENS, embedding_dim=IP_INPUT_DIM, output_dim=768):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim)/dim**.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([nn.ModuleList([
            PerceiverAttention(dim, heads, dim_head),
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim, bias=False),
                          nn.GELU(), nn.Linear(4*dim, dim, bias=False))]) for _ in range(depth)])

    def forward(self, features):
        latents = self.latents.expand(features.shape[0], -1, -1)
        features = self.proj_in(features)
        for attention, feed_forward in self.layers:
            latents = latents+attention(features, latents)
            latents = latents+feed_forward(latents)
        return self.norm_out(self.proj_out(latents))


def read_ip_weights(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Download ip-adapter-plus_sd15.bin and upload to: "+str(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != {"image_proj", "ip_adapter"}:
        raise ValueError("Expected official IP-Adapter Plus SD1.5 checkpoint")
    if tuple(state["image_proj"]["latents"].shape) != (1, 16, 768):
        raise ValueError("Expected Plus SD1.5 Resampler with 16 queries of width 768")
    if tuple(state["image_proj"]["proj_in.weight"].shape) != (768, 1280):
        raise ValueError("Expected ViT-H IP-Adapter Plus, input width 1280")
    return state


def install_ip_attention(unet, pretrained=None, scale=1.):
    from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
    processors = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = AttnProcessor2_0()
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
        elif name.startswith("down_blocks"):
            hidden = unet.config.block_out_channels[int(name.split(".")[1])]
        else:
            raise ValueError("Unknown attention processor path: "+name)
        processors[name] = IPAdapterAttnProcessor2_0(hidden_size=hidden,
            cross_attention_dim=unet.config.cross_attention_dim, num_tokens=(IP_TOKENS,), scale=scale)
    unet.set_attn_processor(processors)
    if pretrained is not None:
        expected = {}
        for index, (name, processor) in enumerate(unet.attn_processors.items()):
            if not isinstance(processor, nn.Module):
                continue
            for key, parameter in processor.state_dict().items():
                # Official checkpoint has a single image adapter; diffusers stores a ModuleList.
                official_key = "{}.{}".format(index, key.replace("to_k_ip.0.", "to_k_ip.")
                                                        .replace("to_v_ip.0.", "to_v_ip."))
                expected[official_key] = parameter
        if set(expected) != set(pretrained):
            raise ValueError("IP attention checkpoint keys do not match this UNet")
        for key, parameter in expected.items():
            if parameter.shape != pretrained[key].shape:
                raise ValueError("IP attention checkpoint shape mismatch: "+key)
        for index, processor in enumerate(unet.attn_processors.values()):
            if isinstance(processor, nn.Module):
                state = {key: pretrained["{}.{}".format(index, key.replace("to_k_ip.0.", "to_k_ip.")
                    .replace("to_v_ip.0.", "to_v_ip."))] for key in processor.state_dict()}
                processor.load_state_dict(state, strict=True)


def image_attention_state(unet):
    return {name: parameter.detach().cpu().contiguous() for name, parameter in unet.named_parameters()
            if ".processor.to_k_ip." in name or ".processor.to_v_ip." in name}


def restore_image_attention(unet, state):
    parameters = {name: parameter for name, parameter in unet.named_parameters()
                  if ".processor.to_k_ip." in name or ".processor.to_v_ip." in name}
    if set(parameters) != set(state):
        raise ValueError("Missing or unexpected trained IP attention keys")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError("Trained IP attention shape mismatch: "+name)
            parameter.copy_(state[name].to(parameter))


def empty_context(model_path, variant=None):
    """The base branch uses a fixed empty-context constant, never user/source text."""
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(model_path), subfolder="tokenizer", local_files_only=True)
    options = {"variant": variant} if variant else {}
    encoder = CLIPTextModel.from_pretrained(str(model_path), subfolder="text_encoder", local_files_only=True, **options)
    encoder.eval().requires_grad_(False)
    ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        return encoder(ids).last_hidden_state.detach().float()
