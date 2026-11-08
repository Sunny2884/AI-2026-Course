#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ROUND="${ROUND:?Set ROUND=1..64; each invocation generates one candidate per source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/teacher_samples/train}"
python "${REPO_ROOT}/teacher/sample_round.py" --model "${MODEL_ROOT}/stable-diffusion-inpainting" \
 --detector "${MODEL_ROOT}/insightface/det_10g.onnx" --data-root "${PROJECT_ROOT}/face_dataset" \
 --csv "${PROJECT_ROOT}/face_dataset/train.csv" --output-root "${OUTPUT_ROOT}" --round "${ROUND}" \
 --resolution "${RESOLUTION:-512}" --steps "${STEPS:-30}" --guidance "${GUIDANCE:-7.5}" \
 --strength "${STRENGTH:-0.85}" --batch-size "${BATCH_SIZE:-4}" \
 --prompt "${PROMPT:-a realistic anonymous portrait, natural face, consistent pose and expression, high quality}" \
 --negative-prompt "${NEGATIVE_PROMPT:-identity resemblance, distorted face, extra eyes, blurry, cartoon, deformed}"
chmod -R 777 "${OUTPUT_ROOT}"
