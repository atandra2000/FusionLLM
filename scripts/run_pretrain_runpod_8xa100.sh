#!/usr/bin/env bash
# Run pre-training on 8× NVIDIA A100 SXM 80GB (RunPod canonical target).
#
# FSDP2 (torch.distributed.fsdp.fully_shard), 8 ranks.  The model +
# optimizer + dataset live in one place: configs/pretrain.yaml.
# WORLD_SIZE=1 falls back to a single-process run for smoke tests.
#
# Usage:
#   bash scripts/run_pretrain_runpod_8xa100.sh
#   WORLD_SIZE=8 bash scripts/run_pretrain_runpod_8xa100.sh
#   EXTRA_ARGS="--resume 5000" bash scripts/run_pretrain_runpod_8xa100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# NCCL tuning for single-node 8×A100 SXM 80GB (NVLink)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
# Higher NCCL queue depth to keep FSDP all-gathers well-fed; cap via
# training.fsdp_limit_all_gathers in the YAML.
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"

WORLD_SIZE="${WORLD_SIZE:-8}"
CONFIG="${CONFIG:-configs/pretrain.yaml}"

echo "RunPod 8×A100 SXM 80GB launcher"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  WORLD_SIZE=${WORLD_SIZE}"
echo "  CONFIG=${CONFIG}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
  exec python training/pretrain.py \
    --config "$CONFIG" \
    "${@:-${EXTRA_ARGS:-}}"
else
  exec torchrun \
    --nproc_per_node="${WORLD_SIZE}" \
    --nnodes="${NNODES:-1}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    training/pretrain.py \
    --config "$CONFIG" \
    "${@:-${EXTRA_ARGS:-}}"
fi
