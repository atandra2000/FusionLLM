# models/moe/dispatch.py
"""Dispatch strategies for MoE routing.

Provides scatter-gather, Triton grouped-GEMM, and all-to-all dispatch.
Each strategy takes the same arguments so they can be used interchangeably.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def scatter_gather_dispatch(
    flat: torch.Tensor,
    flat_token_ids_sorted: torch.Tensor,
    flat_weights_sorted: torch.Tensor,
    expert_start: torch.Tensor,
    expert_size: torch.Tensor,
    active_indices: torch.Tensor,
    y_routed: torch.Tensor,
    expert_w1_stack: torch.Tensor,
    expert_w2_stack: torch.Tensor,
    expert_w3_stack: torch.Tensor | None,
    activation: str,
    dim: int,
    expert_forward_fn,
) -> torch.Tensor:
    """Scatter-gather dispatch: iterate active experts, compute, scatter-add."""
    active_list = active_indices.tolist()
    if len(active_list) == 0:
        return y_routed

    w1_stack = expert_w1_stack[active_list]
    w2_stack = expert_w2_stack[active_list]
    w3_stack = expert_w3_stack[active_list] if expert_w3_stack is not None else None

    active_starts = expert_start[active_indices]
    active_sizes = expert_size[active_indices]

    y_routed_fp32 = y_routed.float()

    for i, local_idx in enumerate(active_list):
        start = active_starts[i]
        size = active_sizes[i]
        token_ids = flat_token_ids_sorted[start : start + size]
        w = flat_weights_sorted[start : start + size].unsqueeze(-1)
        expert_out = expert_forward_fn(
            flat[token_ids], w1_stack[i], w2_stack[i],
            w3_stack[i] if w3_stack is not None else None,
        )
        y_routed_fp32.scatter_add_(
            0,
            token_ids.unsqueeze(-1).expand(-1, dim),
            expert_out * w,
        )
    y_routed.copy_(y_routed_fp32.to(y_routed.dtype))
    return y_routed


def try_grouped_gemm(
    flat: torch.Tensor,
    flat_token_ids_sorted: torch.Tensor,
    flat_weights_sorted: torch.Tensor,
    expert_start: torch.Tensor,
    expert_size: torch.Tensor,
    active_indices: torch.Tensor,
    y_routed: torch.Tensor,
    expert_w1_stack: torch.Tensor,
    expert_w2_stack: torch.Tensor,
    expert_w3_stack: torch.Tensor | None,
    activation: str,
) -> bool:
    """Try the Triton grouped-GEMM fast-path.  Returns True on success."""
    if active_indices.numel() == 0:
        return False
    try:
        from ops.triton.grouped_gemm import grouped_gemm, has_triton

        if not has_triton():
            return False

        n_active = active_indices.size(0)
        offsets_list = [0]
        token_gather: list[int] = []
        for i_active in range(n_active):
            idx = active_indices[i_active].item()
            start = expert_start[idx].item()
            size = expert_size[idx].item()
            if size > 0:
                offsets_list.append(offsets_list[-1] + size)
                token_ids = flat_token_ids_sorted[start: start + size].tolist()
                token_gather.extend(token_ids)
        if len(token_gather) == 0:
            return False

        offsets = torch.tensor(offsets_list, dtype=torch.int32, device=flat.device)
        a_grouped = flat[token_gather]

        active_list = active_indices.tolist()
        w1_stack = expert_w1_stack[active_list]
        w2_stack = expert_w2_stack[active_list]

        inter = grouped_gemm(a_grouped, w1_stack, offsets)
        if activation == "swiglu":
            w3_stack = expert_w3_stack[active_list]
            gate = grouped_gemm(a_grouped, w3_stack, offsets)
            h = torch.silu(inter) * gate
        else:
            h = torch.relu(inter).square()
        out_grouped = grouped_gemm(h, w2_stack, offsets)

        y_routed_32 = y_routed.float()
        for i_active in range(n_active):
            idx = active_indices[i_active].item()
            start = expert_start[idx].item()
            size = expert_size[idx].item()
            if size == 0:
                continue
            o_start = offsets[i_active].item()
            o_end = offsets[i_active + 1].item()
            token_ids = flat_token_ids_sorted[start: start + size]
            w = flat_weights_sorted[start: start + size].unsqueeze(-1)
            y_routed_32.scatter_add_(
                0,
                token_ids.unsqueeze(-1).expand(-1, flat.size(1)),
                out_grouped[o_start:o_end] * w,
            )
        y_routed.copy_(y_routed_32.to(y_routed.dtype))
        return True
    except (ImportError, NotImplementedError):
        return False


def all_to_all_dispatch(
    flat: torch.Tensor,
    flat_token_ids_sorted: torch.Tensor,
    flat_weights_sorted: torch.Tensor,
    expert_start: torch.Tensor,
    expert_size: torch.Tensor,
    active_indices: torch.Tensor,
    y_routed: torch.Tensor,
    expert_w1_stack: torch.Tensor,
    expert_w2_stack: torch.Tensor,
    expert_w3_stack: torch.Tensor | None,
    activation: str,
    dim: int,
    world_size: int,
    expert_forward_fn,
) -> torch.Tensor:
    """All-to-all expert dispatch (DeepSeek-V3 style) - falls back to scatter-gather.

    TODO: Full all-to-all dispatch (DeepSeek-V3 style).
    For now, reuse the same scatter-gather path via try_grouped_gemm.
    """
    import torch.distributed as dist

    active_list = active_indices.tolist()

    # Try grouped GEMM first
    if len(active_list) > 0 and try_grouped_gemm(
        flat, flat_token_ids_sorted, flat_weights_sorted,
        expert_start, expert_size, active_indices, y_routed,
        expert_w1_stack, expert_w2_stack, expert_w3_stack, activation,
    ):
        pass  # grouped GEMM handled it
    elif len(active_list) > 0:
        # Fall back to scatter-gather
        scatter_gather_dispatch(
            flat, flat_token_ids_sorted, flat_weights_sorted,
            expert_start, expert_size, active_indices, y_routed,
            expert_w1_stack, expert_w2_stack, expert_w3_stack,
            activation, dim, expert_forward_fn,
        )

    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(y_routed, op=dist.ReduceOp.SUM)

    return y_routed
