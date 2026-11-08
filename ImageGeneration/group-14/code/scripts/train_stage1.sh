#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SEED="${SEED:-1324400183}"
export MODEL="${MODEL:-${MODEL_ROOT}/realistic-vision-v5.1}"
export TRAINING_VAE="${TRAINING_VAE:-${MODEL}/vae}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PROJECT_ROOT}/checkpoints/stage1_original_vae}"
# Stage 1 matches the historical v4 validation protocol.
export EVAL_SAMPLER=ddim EVAL_IP_SCALE=1.0 EVAL_GUIDANCE_SCALE=3.0
export EVAL_VAE="${TRAINING_VAE}"
bash "${REPO_ROOT}/scripts/train.sh"
chmod -R 777 "${PROJECT_ROOT}/checkpoints"
