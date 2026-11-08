#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GENERATED="${GENERATED:?Set GENERATED to the complete test generation directory}"
SUBMISSION="${SUBMISSION:-${PROJECT_ROOT}/outputs/submission.csv}"
mkdir -p "$(dirname "${SUBMISSION}")"
python "${REPO_ROOT}/scripts/make_submission.py" "${GENERATED}" --test "${PROJECT_ROOT}/face_dataset/test.csv" --output "${SUBMISSION}"
chmod 777 "${SUBMISSION}"
