#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a complete v5 epoch checkpoint}"
MODEL="${MODEL:-${MODEL_ROOT}/realistic-vision-v5.1}"
CONDITIONS="${CONDITIONS:?Set CONDITIONS to the CLIP-only cached feature manifest}"
RUN_ID="${RUN_ID:-$(python run_id.py)}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/clip_only_cfg_v4/infer_${RUN_ID}}"
SEED="${SEED:-$(python -c 'import secrets; print(secrets.randbits(32))')}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
LIMIT="${LIMIT:-0}"
SAMPLER="${SAMPLER:-dpmpp_2m_karras}"
VAE="${VAE:-${MODEL_ROOT}/sd-vae-ft-mse}"
IP_SCALE="${IP_SCALE:-0.8}"
set --
if [[ -n "${VAE}" ]]; then set -- "$@" --vae "${VAE}"; fi
if [[ -n "${IP_SCALE}" ]]; then set -- "$@" --ip-scale "${IP_SCALE}"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python infer.py --model "${MODEL}" --checkpoint "${CHECKPOINT}" --conditions "${CONDITIONS}" \
  --output "${OUTPUT}" --run-id "${RUN_ID}" --seed "${SEED}" --batch-size "${BATCH_SIZE}" --steps "${STEPS}" --limit "${LIMIT}" --guidance-scale "${GUIDANCE_SCALE}" --sampler "${SAMPLER}" "$@"
chmod -R 777 "${OUTPUT}"
