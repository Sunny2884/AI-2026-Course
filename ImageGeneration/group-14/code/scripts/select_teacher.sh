#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPUS="${GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/data/selected64_v1}"
torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS}" "${REPO_ROOT}/data_selection/select_candidates.py" \
 --project-root "${PROJECT_ROOT}" --model-root "${MODEL_ROOT}" --output "${OUTPUT}" \
 --rounds "${ROUNDS:-64}" --batch-size "${BATCH_SIZE:-8}" --io-workers "${IO_WORKERS:-4}" --resume
python "${REPO_ROOT}/data_selection/merge.py" --output "${OUTPUT}" \
 --train-csv "${PROJECT_ROOT}/face_dataset/train.csv" --data-root "${PROJECT_ROOT}/face_dataset"
chmod -R 777 "${OUTPUT}"
