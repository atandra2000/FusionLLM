# utils/distributed.py
"""
Distributed training utilities for the project.

Identity
--------
* **FSDP2 only** (``torch.distributed.fsdp.fully_shard``).  DDP is
  intentionally not supported; the canonical target is a single
  8×A100 SXM 80GB RunPod node, but the helpers also work for multi-node
  via the standard ``torchrun`` env-vars.
* NCCL backend for GPU collectives; bf16 reduced precision for
  FSDP2 all-gathers.
* Async checkpointing and async W&B logging are owned by
  their respective modules; this file is for setup + collectives.

The FSDP2 wrapping policy is conservative: per-TransformerBlock
auto-wrap, ``FULL_SHARD`` strategy, backward prefetch on, forward
prefetch off, ``limit_all_gathers=True``.  Multi-node runs need
``BACKEND_PRE_HOOK`` env-vars from the torchrun launcher.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# ── NCCL Profiling ──────────────────────────────────────────────────────────

@dataclass
class CommProfile:
    """Profile result for a single communication operation."""
    operation: str
    latency_ms: float
    message_bytes: int
    timestamp: float


class NCCLProfiler:
    """Lightweight NCCL communication profiler.
    
    Tracks latency and message sizes for communication operations.
    Enabled via FUSIONLLM_PROFILE_COMMS=1 environment variable.
    """
    
    _instance: Optional["NCCLProfiler"] = None
    
    def __init__(self):
        self.enabled = os.environ.get("FUSIONLLM_PROFILE_COMMS", "0") == "1"
        self.profiles: List[CommProfile] = []
        self._active: Dict[str, float] = {}
    
    @classmethod
    def get_instance(cls) -> "NCCLProfiler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @contextmanager
    def profile(self, operation: str, message_bytes: int = 0):
        """Profile a communication operation."""
        if not self.enabled or not dist.is_initialized():
            yield
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        self.profiles.append(CommProfile(
            operation=operation,
            latency_ms=elapsed,
            message_bytes=message_bytes,
            timestamp=time.time(),
        ))
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics per operation type."""
        if not self.profiles:
            return {}
        
        summary = {}
        for p in self.profiles:
            if p.operation not in summary:
                summary[p.operation] = {
                    "count": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "total_bytes": 0,
                }
            s = summary[p.operation]
            s["count"] += 1
            s["total_ms"] += p.latency_ms
            s["max_ms"] = max(s["max_ms"], p.latency_ms)
            s["total_bytes"] += p.message_bytes
        
        for op, s in summary.items():
            s["avg_ms"] = s["total_ms"] / s["count"] if s["count"] > 0 else 0
            s["total_bytes_gb"] = s["total_bytes"] / (1024**3)
        
        return summary
    
    def print_summary(self):
        """Print profiling summary."""
        if not self.enabled:
            return
        
        summary = self.get_summary()
        if not summary:
            return
        
        print(f"\n{'='*60}")
        print("NCCL Communication Profile")
        print(f"{'='*60}")
        for op, stats in sorted(summary.items()):
            print(f"{op}:")
            print(f"  Count:     {stats['count']}")
            print(f"  Avg:       {stats['avg_ms']:.3f} ms")
            print(f"  Max:       {stats['max_ms']:.3f} ms")
            print(f"  Total:     {stats['total_ms']:.3f} ms")
            print(f"  Data:      {stats['total_bytes_gb']:.3f} GB")
        print(f"{'='*60}\n")
    
    def reset(self):
        """Reset all profiles."""
        self.profiles.clear()


# ── Setup / teardown ────────────────────────────────────────────────────────


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """
    Initialize the distributed process group.

    Returns:
        (world_size, rank, local_rank)
    """
    if not dist.is_available():
        print("[distributed] torch.distributed not available — single-GPU mode")
        return (1, 0, 0)

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, init_method="env://")
            print(
                f"[distributed] rank={dist.get_rank()}/{dist.get_world_size()} "
                f"local_rank={local_rank} backend={backend}"
            )
        else:
            print("[distributed] world_size=1 — single-GPU mode (FSDP no-op)")
            return (1, 0, 0)

    return (
        dist.get_world_size(),
        dist.get_rank(),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def cleanup_distributed() -> None:
    """Tear down the distributed process group (no-op if not initialised)."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Collective helpers ──────────────────────────────────────────────────────


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and return its mean across all ranks."""
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    with NCCLProfiler.get_instance().profile("all_reduce", tensor.nelement() * tensor.element_size()):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """All-gather a tensor from all ranks and return a list."""
    if not dist.is_initialized():
        return [tensor]
    world_size = dist.get_world_size()
    output = [torch.empty_like(tensor) for _ in range(world_size)]
    with NCCLProfiler.get_instance().profile("all_gather", tensor.nelement() * tensor.element_size()):
        dist.all_gather(output, tensor)
    return output


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_to_all_single(
    output: torch.Tensor, input: torch.Tensor, group=None
) -> None:
    """All-to-all single tensor operation for expert parallelism.
    
    Args:
        output: Pre-allocated output tensor
        input: Input tensor to scatter
        group: Process group (default: WORLD)
    """
    if not dist.is_initialized():
        output.copy_(input)
        return
    if group is None:
        group = dist.group.WORLD
    dist.all_to_all_single(output, input, group=group)


def all_to_all(output_list: list, input_list: list, group=None) -> None:
    """All-to-all list of tensors operation.
    
    Args:
        output_list: Pre-allocated list of output tensors
        input_list: List of input tensors to scatter
        group: Process group (default: WORLD)
    """
    if not dist.is_initialized():
        output_list[0].copy_(input_list[0])
        return
    if group is None:
        group = dist.group.WORLD
    dist.all_to_all(output_list, input_list, group=group)


# ── FSDP2 wrapping helpers ──────────────────────────────────────────────────


def wrap_fsdp2(
    model: nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    fsdp_shard_strategy: str = "FULL_SHARD",
    fsdp_forward_prefetch: bool = False,
    fsdp_backward_prefetch: bool = True,
    limit_all_gathers: bool = True,
) -> nn.Module:
    """
    Apply FSDP2 (``fully_shard``) to ``model``.

    The wrapping policy is *per-TransformerBlock* auto-wrap — a good
    default for hybrid MLA + Mamba-2 + MoE backbones.  Expert linears
    inside DeepSeekMoE are sharded as part of the surrounding block;
    no per-expert wrapping is attempted (the experts are small enough
    that a per-block wrap shards them all uniformly).

    Args:
        model: the raw (unwrapped) model — typically the result of
               ``Transformer(...)``.
        param_dtype: dtype for sharded parameters (default bf16).
        reduce_dtype: dtype for gradient reductions (default fp32).
        fsdp_shard_strategy: ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``, or
                             ``"NO_SHARD"``.
        fsdp_forward_prefetch: forward prefetch (default off — saves
                               H2D bandwidth on the hot path).
        fsdp_backward_prefetch: backward prefetch (default on — the
                                default since PyTorch 2.4).
        limit_all_gathers: cap NCCL queue depth (default on).
    """
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        MixedPrecisionPolicy,
        ShardingStrategy,
        fully_shard,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    if not dist.is_initialized() or dist.get_world_size() == 1:
        print("[fsdp2] world_size=1 — skipping fully_shard wrap (single-GPU run)")
        return model

    # ── Auto-wrap policy: one FSDP unit per TransformerBlock ────────────
    # Defer the import to avoid a hard dep on the model's block class.
    block_cls = None
    for m in model.modules():
        if m.__class__.__name__ == "TransformerBlock":
            block_cls = m.__class__
            break
    if block_cls is None:
        print(
            "[fsdp2] warning: no TransformerBlock found — falling back to root "
            "wrap (single fully_shard on the entire model)"
        )
        wrap_policy = None
    else:
        wrap_policy = ModuleWrapPolicy({block_cls})

    # ── Sharding strategy ───────────────────────────────────────────────
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    if fsdp_shard_strategy not in strategy_map:
        raise ValueError(
            f"Unknown FSDP shard strategy: {fsdp_shard_strategy!r}. "
            f"Choose from {list(strategy_map)}."
        )
    sharding_strategy = strategy_map[fsdp_shard_strategy]

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    # ── Backward prefetch ───────────────────────────────────────────────
    bwd_prefetch = (
        BackwardPrefetch.BACKWARD_PRE if fsdp_backward_prefetch else BackwardPrefetch.BACKWARD_POST
    )

    # ── Inner wrap: apply fully_shard to every TransformerBlock ────────
    if wrap_policy is not None:
        for module in model.modules():
            if isinstance(module, block_cls):
                fully_shard(
                    module,
                    mp_policy=mp_policy,
                    sharding_strategy=sharding_strategy,
                    backward_prefetch=bwd_prefetch,
                    limit_all_gathers=limit_all_gathers,
                )

    # ── Outer wrap: one fully_shard on the root ─────────────────────────
    fully_shard(
        model,
        mp_policy=mp_policy,
        sharding_strategy=sharding_strategy,
        backward_prefetch=bwd_prefetch,
        limit_all_gathers=limit_all_gathers,
    )

    if is_main_process():
        print(
            f"[fsdp2] wrapped: strategy={fsdp_shard_strategy} "
            f"param_dtype={param_dtype} reduce_dtype={reduce_dtype} "
            f"bwd_prefetch={bool(fsdp_backward_prefetch)} "
            f"limit_all_gathers={limit_all_gathers}"
        )

    return model


# ── Per-layer reshard tuning (Phase 3.2) ────────────────────────────────


def configure_reshard(
    model: nn.Module,
    keep_last_n: int = 1,
) -> None:
    """Configure ``reshard_after_forward`` per FSDP unit.

    Keeps parameter shards resident after the forward pass on the last
    ``keep_last_n`` FSDP units.  All earlier units reshard (free params)
    as usual.

    Keeping the last N units resident avoids an all-gather on the first
    backward pass through those layers, at the cost of higher peak memory
    during the backward of earlier layers that still reference those
    parameters.  ``keep_last_n=1`` is a safe default for most models.

    This is a no-op when no FSDP units are found (e.g. single-GPU runs).
    """
    fsdp_units: list[nn.Module] = []
    for module in model.modules():
        if hasattr(module, "reshard_after_forward"):
            fsdp_units.append(module)

    n_units = len(fsdp_units)
    if n_units == 0:
        return

    for i, unit in enumerate(fsdp_units):
        unit.reshard_after_forward = i < n_units - keep_last_n

    if is_main_process() and n_units > 0:
        print(
            f"[fsdp2] reshard config: {n_units} units, "
            f"keeping last {keep_last_n} resident "
            f"(units {n_units - keep_last_n}–{n_units - 1})"
        )
