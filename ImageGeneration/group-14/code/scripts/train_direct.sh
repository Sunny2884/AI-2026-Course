#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export TRAINING_VAE="${TRAINING_VAE:-${MODEL_ROOT}/sd-vae-ft-mse}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PROJECT_ROOT}/checkpoints/direct_mse}"
bash "${REPO_ROOT}/scripts/train.sh"
chmod -R 777 "${PROJECT_ROOT}/checkpoints"
