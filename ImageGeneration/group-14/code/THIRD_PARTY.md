# Third-party components

The Resampler layout in `src/ip_adapter.py` follows Tencent AI Lab IP-Adapter (Apache-2.0), adapted to PyTorch SDPA; source: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py . Preserve this attribution and the upstream license when redistributing derived portions: https://github.com/tencent-ailab/IP-Adapter/blob/main/LICENSE .

This source bundle does not redistribute datasets, model weights, upstream packages or OpenFace binaries. Each dependency, model and dataset retains its own license. In particular, consult Realistic Vision/Stable Diffusion model cards, CLIP/IP-Adapter licenses, FFHQ terms, InsightFace pretrained-model restrictions, HSEmotion/MUSIQ/OpenFace licenses before using or distributing those assets. No license for third-party assets is implied by this code bundle.

The full upstream Apache-2.0 license is included in `licenses/IP-Adapter-Apache-2.0.txt`. The adapted Resampler uses PyTorch SDPA instead of einops reshaping. The upstream Resampler credits OpenFlamingo helpers and lucidrains/imagen-pytorch as sources; retain those source references when redistributing: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py .
