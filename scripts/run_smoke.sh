#!/usr/bin/env bash
# Smoke test: 1 GPU, 10 optimisation steps, synthetic data, tiny model.
# Usage:
#   bash scripts/run_smoke.sh
#   CONFIG=configs/my_smoke.yaml bash scripts/run_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/smoke_pretrain.yaml}"

echo "===== Smoke test (1 GPU) ====="
echo "  CONFIG=${CONFIG}"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

exec python training/pretrain.py \
  --config "$CONFIG" \
  "${@:-}"
