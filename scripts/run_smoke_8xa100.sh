#!/usr/bin/env bash
# Smoke test: 8 GPUs, 10 optimisation steps, synthetic data, tiny model.
# Usage:
#   bash scripts/run_smoke_8xa100.sh
#   CONFIG=configs/my_smoke.yaml bash scripts/run_smoke_8xa100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"

CONFIG="${CONFIG:-configs/smoke_pretrain.yaml}"
WORLD_SIZE="${WORLD_SIZE:-8}"

echo "===== Smoke test (${WORLD_SIZE} GPUs) ====="
echo "  CONFIG=${CONFIG}"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

exec torchrun \
  --nproc_per_node="${WORLD_SIZE}" \
  --nnodes="${NNODES:-1}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  training/pretrain.py \
  --config "$CONFIG" \
  "${@:-}"
