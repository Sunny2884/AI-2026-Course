#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/train_single_gpu.sh GPU_ID [additional train.py arguments]" >&2
  exit 2
fi
GPU_ID="$1"
shift
export CUDA_VISIBLE_DEVICES="$GPU_ID"
RUN_OUTPUT="outputs/rgbd_single_gpu_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
exec python train.py --num-gpus 1 --device cuda --output "$RUN_OUTPUT" "$@"
