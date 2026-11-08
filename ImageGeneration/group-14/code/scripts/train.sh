#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL="${MODEL:-${MODEL_ROOT}/realistic-vision-v5.1}"
MODEL_VARIANT="${MODEL_VARIANT:-}"
IMAGE_ENCODER="${IMAGE_ENCODER:-${MODEL_ROOT}/ip_adapter/models/image_encoder}"
REUSE_FEATURES_ROOT="${REUSE_FEATURES_ROOT:-${PROJECT_ROOT}/data/features_v3}"
CONDITION_DROPOUT="${CONDITION_DROPOUT:-0.1}"
IP_ADAPTER_WEIGHT="${IP_ADAPTER_WEIGHT:-${MODEL_ROOT}/ip_adapter/ip-adapter-plus_sd15.bin}"
IP_SCALE="${IP_SCALE:-1.0}"
TRAINING_VAE="${TRAINING_VAE:-${MODEL_ROOT}/sd-vae-ft-mse}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
if [[ -n "${INIT_CHECKPOINT}" && -n "${RESUME:-}" ]]; then echo "Use INIT_CHECKPOINT or RESUME, not both" >&2; exit 1; fi
if [[ ! -f "${IP_ADAPTER_WEIGHT}" ]]; then
  echo "Missing IP-Adapter Plus weight: ${IP_ADAPTER_WEIGHT}" >&2
  exit 1
fi
PAIRS="${PAIRS:-${PROJECT_ROOT}/data/selected64_v1/conditions.jsonl}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/face_dataset}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/train.csv}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PROJECT_ROOT}/checkpoints/clip_mse_v5}"
RESUME="${RESUME:-}"
set --
if [[ -n "${RESUME}" ]]; then set -- --resume "${RESUME}"; fi
RUN_ID="${RUN_ID:-$(python run_id.py "$@")}"
if [[ -n "${RESUME}" ]]; then
  OUTPUT="${OUTPUT:-$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent)' "${RESUME}")}"
else
  OUTPUT="${OUTPUT:-${OUTPUT_PREFIX}_${RUN_ID}}"
fi
export RUN_ID OUTPUT
echo "RUN_ID=${RUN_ID}"
echo "OUTPUT=${OUTPUT}"
GPUS="${GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
if [[ "${GPUS}" -lt 1 || $((GLOBAL_BATCH % (BATCH_SIZE * GPUS))) -ne 0 ]]; then
  echo "GLOBAL_BATCH must be divisible by BATCH_SIZE * GPUS; CUDA GPUs must be visible" >&2
  exit 1
fi
ACCUMULATION="${ACCUMULATION:-$((GLOBAL_BATCH / (BATCH_SIZE * GPUS)))}"
EPOCHS="${EPOCHS:-10}"
RESOLUTION="${RESOLUTION:-512}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
ADAPTER_LEARNING_RATE="${ADAPTER_LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
FEATURES_ROOT="${FEATURES_ROOT:-${PROJECT_ROOT}/data/features_v4}"
FEATURES_OUTPUT="${FEATURES_OUTPUT:-${FEATURES_ROOT}/train}"
PREPARE_FEATURES="${PREPARE_FEATURES:-1}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}"
FEATURE_WORKERS="${FEATURE_WORKERS:-4}"
FEATURE_LIMIT="${FEATURE_LIMIT:-0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-20261031}"
MAX_STEPS="${MAX_STEPS:-0}"
PRECISION="${PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-10}"
HISTOGRAM_EVERY="${HISTOGRAM_EVERY:-100}"
IMAGE_EVERY="${IMAGE_EVERY:-100}"
AUTO_EVAL="${AUTO_EVAL:-1}"
EVAL_EPOCHS="${EVAL_EPOCHS:-all}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_STEPS="${EVAL_STEPS:-30}"
EVAL_SEED="${EVAL_SEED:-20261026}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_GUIDANCE_SCALE="${EVAL_GUIDANCE_SCALE:-3.0}"
if [[ "${AUTO_EVAL}" != 0 && "${AUTO_EVAL}" != 1 ]]; then
  echo "AUTO_EVAL must be 0 or 1" >&2
  exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
set --
if [[ -n "${RESUME}" ]]; then set -- --resume "${RESUME}"; fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then set -- "$@" --init-checkpoint "${INIT_CHECKPOINT}"; fi
python base_model.py --model "${MODEL}" --variant "${MODEL_VARIANT}"

if [[ "${PREPARE_FEATURES}" == 1 ]] && ! python check_cache.py --cache "${FEATURES_OUTPUT}" --csv "${TRAIN_CSV}" --pairs "${PAIRS}" --data-root "${DATA_ROOT}" --limit "${FEATURE_LIMIT}"; then
  torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS}" prepare_features.py \
    --csv "${TRAIN_CSV}" --pairs "${PAIRS}" --data-root "${DATA_ROOT}" \
    --reuse-cache "${REUSE_FEATURES_ROOT}/train" --image-encoder "${IMAGE_ENCODER}" \
    --output "${FEATURES_OUTPUT}" --batch-size "${FEATURE_BATCH_SIZE}" --workers "${FEATURE_WORKERS}" \
    --limit "${FEATURE_LIMIT}" --device cuda
elif [[ "${PREPARE_FEATURES}" != 0 && "${PREPARE_FEATURES}" != 1 ]]; then
  echo "PREPARE_FEATURES must be 0 or 1" >&2
  exit 1
fi
python check_cache.py --cache "${FEATURES_OUTPUT}" --csv "${TRAIN_CSV}" --pairs "${PAIRS}" \
  --data-root "${DATA_ROOT}" --limit "${FEATURE_LIMIT}"
if [[ "${AUTO_EVAL}" == 1 && "${PREPARE_FEATURES}" == 1 ]] && ! python check_cache.py --cache "${EVAL_FEATURES_CACHE:-${FEATURES_ROOT}/val}" --csv "${DATA_ROOT}/val.csv" --data-root "${DATA_ROOT}" --requested-limit "${EVAL_LIMIT}"; then
  PROJECT_ROOT="${PROJECT_ROOT}" DATA_ROOT="${DATA_ROOT}" IMAGE_ENCODER="${IMAGE_ENCODER}" REUSE_FEATURES_ROOT="${REUSE_FEATURES_ROOT}" \
    SPLIT=val OUTPUT="${EVAL_FEATURES_CACHE:-${FEATURES_ROOT}/val}" LIMIT="${EVAL_LIMIT}" BATCH_SIZE="${FEATURE_BATCH_SIZE}" GPUS="${GPUS}" bash "${REPO_ROOT}/scripts/prepare.sh"
fi
if [[ "${AUTO_EVAL}" == 1 ]]; then
  python check_cache.py --cache "${EVAL_FEATURES_CACHE:-${FEATURES_ROOT}/val}" \
    --csv "${DATA_ROOT}/val.csv" --data-root "${DATA_ROOT}" --requested-limit "${EVAL_LIMIT}"
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS}" train.py \
  --model "${MODEL}" --pairs "${FEATURES_OUTPUT}/conditions.jsonl" --data-root "${DATA_ROOT}" --train-csv "${TRAIN_CSV}" \
  --output "${OUTPUT}" --run-id "${RUN_ID}" --resolution "${RESOLUTION}" --batch-size "${BATCH_SIZE}" \
  --training-vae "${TRAINING_VAE}" --ip-adapter-weight "${IP_ADAPTER_WEIGHT}" --ip-scale "${IP_SCALE}" \
  --model-variant "${MODEL_VARIANT}" --condition-dropout "${CONDITION_DROPOUT}" \
  --accumulation "${ACCUMULATION}" --epochs "${EPOCHS}" --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}" --learning-rate "${LEARNING_RATE}" \
  --adapter-learning-rate "${ADAPTER_LEARNING_RATE}" --weight-decay "${WEIGHT_DECAY}" \
  --warmup-ratio "${WARMUP_RATIO}" --max-grad-norm "${MAX_GRAD_NORM}" \
  --workers "${WORKERS}" --seed "${SEED}" --max-steps "${MAX_STEPS}" --mixed-precision "${PRECISION}" \
  --log-every "${LOG_EVERY}" --histogram-every "${HISTOGRAM_EVERY}" --image-every "${IMAGE_EVERY}" "$@"
PERMISSION_PATHS=("${OUTPUT}" "${FEATURES_OUTPUT}")
if [[ "${AUTO_EVAL}" == 1 ]]; then
  EVAL_OUTPUT="${EVAL_OUTPUT:-${PROJECT_ROOT}/outputs/evaluation/$(basename "${OUTPUT}")_val_$(python run_id.py)}"
  echo "Training complete; evaluating checkpoints ${EVAL_EPOCHS} from ${OUTPUT}"
  PROJECT_ROOT="${PROJECT_ROOT}" MODEL_ROOT="${MODEL_ROOT}" DATA_ROOT="${DATA_ROOT}" RUN_DIR="${OUTPUT}" \
    FEATURES_CACHE="${EVAL_FEATURES_CACHE:-${FEATURES_ROOT}/val}" \
    VAE="${TRAINING_VAE}" SAMPLER="${EVAL_SAMPLER:-dpmpp_2m_karras}" IP_SCALE="${EVAL_IP_SCALE:-0.8}" GUIDANCE_SCALE="${EVAL_GUIDANCE_SCALE}" \
    OUTPUT="${EVAL_OUTPUT}" SPLIT=val EPOCHS="${EVAL_EPOCHS}" BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    LIMIT="${EVAL_LIMIT}" STEPS="${EVAL_STEPS}" SEED="${EVAL_SEED}" DEVICE="${EVAL_DEVICE}" \
    bash "${REPO_ROOT}/scripts/evaluate.sh"
  PERMISSION_PATHS+=("${EVAL_OUTPUT}")
  echo "Training and evaluation complete; metrics: ${EVAL_OUTPUT}/metrics.csv"
fi
chmod -R 777 "${PERMISSION_PATHS[@]}"
