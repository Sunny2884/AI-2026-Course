#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the training run directory containing epoch_XXX}"
EPOCHS="${EPOCHS:-all}"
SPLIT="${SPLIT:-val}"
RUN_ID="${RUN_ID:-$(python run_id.py)}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/evaluation/$(basename "${RUN_DIR}")_${SPLIT}_${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/face_dataset}"
EVALUATION_CODE="${EVALUATION_CODE:-${REPO_ROOT}/evaluation/evaluate.py}"
SCORING_CODE="${SCORING_CODE:-${REPO_ROOT}/data_selection/scoring.py}"
OPENFACE_BIN="${OPENFACE_BIN:-${OPENFACE_ROOT}/build/bin/FaceLandmarkImg}"
SKIP_OPENFACE="${SKIP_OPENFACE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-30}"
SEED="${SEED:-20261026}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-4}"
VAE="${VAE:-${MODEL_ROOT}/sd-vae-ft-mse}"
SAMPLER="${SAMPLER:-dpmpp_2m_karras}"
IP_SCALE="${IP_SCALE:-0.8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
FEATURES_CACHE="${FEATURES_CACHE:-${PROJECT_ROOT}/data/features_v4/${SPLIT}}"
export LD_LIBRARY_PATH="${OPENFACE_ROOT}/install/lib64:${OPENFACE_ROOT}/install/lib:${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${THREADS}"
set --
if [[ "${SKIP_OPENFACE}" == 1 ]]; then set -- --skip-openface; fi
echo "Evaluation output: ${OUTPUT}"
python evaluate_checkpoints.py --run "${RUN_DIR}" --epochs "${EPOCHS}" --split "${SPLIT}" \
  --data-root "${DATA_ROOT}" --model-root "${MODEL_ROOT}" --evaluation-code "${EVALUATION_CODE}" \
  --scoring-code "${SCORING_CODE}" --output "${OUTPUT}" --openface-bin "${OPENFACE_BIN}" \
  --seed "${SEED}" --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --limit "${LIMIT}" \
  --device "${DEVICE}" --threads "${THREADS}" --features-cache "${FEATURES_CACHE}" --guidance-scale "${GUIDANCE_SCALE}" --vae "${VAE}" --sampler "${SAMPLER}" --ip-scale "${IP_SCALE}" "$@"
chmod -R 777 "${OUTPUT}"
