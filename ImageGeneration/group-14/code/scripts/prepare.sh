#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
IMAGE_ENCODER="${IMAGE_ENCODER:-${MODEL_ROOT}/ip_adapter/models/image_encoder}"
REUSE_FEATURES_ROOT="${REUSE_FEATURES_ROOT:-${PROJECT_ROOT}/data/features_v3}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/face_dataset}"
SPLIT="${SPLIT:-train}"
CSV="${CSV:-${DATA_ROOT}/${SPLIT}.csv}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/data/features_v4/${SPLIT}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
GPUS="${GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
LIMIT="${LIMIT:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
set --
if [[ "${SPLIT}" == train ]]; then
  PAIRS="${PAIRS:-${PROJECT_ROOT}/data/selected64_v1/conditions.jsonl}"
  set -- --pairs "${PAIRS}"
fi
torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS}" prepare_features.py \
  --csv "${CSV}" --data-root "${DATA_ROOT}" --reuse-cache "${REUSE_FEATURES_ROOT}/${SPLIT}" \
  --image-encoder "${IMAGE_ENCODER}" --output "${OUTPUT}" \
  --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" --limit "${LIMIT}" --device cuda "$@"
chmod -R 777 "${OUTPUT}"
