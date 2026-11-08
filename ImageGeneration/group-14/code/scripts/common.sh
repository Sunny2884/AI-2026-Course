#!/usr/bin/env bash
# Source from any script; execution does not depend on the caller's cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CONDA_ENV:-}" ]]; then
  set +u
  source activate "${CONDA_ENV}"
fi
set -u
export REPO_ROOT
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/workspace}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export OPENFACE_ROOT="${OPENFACE_ROOT:-${PROJECT_ROOT}/tools/openface}"
cd "${REPO_ROOT}/src"
