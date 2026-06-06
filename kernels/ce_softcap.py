# kernels/ce_softcap.py
"""Fused cross-entropy loss with built-in logit softcap.

The softcap operation (modded-nanogpt #18) clamps logits to
``[-softcap, softcap]`` before the softmax:

    logits = softcap * tanh(logits / softcap)

A fused version avoids materialising the clamped gradient when the
softcap is applied during the backward pass.  This file provides:

* :func:`ce_softcap` — a pure-PyTorch fallback that applies the softcap
  then calls :func:`F.cross_entropy`.
* :func:`fused_ce_softcap` — the fused Triton kernel (available when
  CUDA + triton are present).

Reference
---------
Keller Jordan, "modded-nanogpt speedrun", record #79 (April 2026).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


def softcap(logits: torch.Tensor, softcap_value: float = 15.0) -> torch.Tensor:
    """Apply the tanh-based logit softcap in-place.

    ``out = softcap_value * tanh(logits / softcap_value)``

    Args:
        logits: input logits (any shape).
        softcap_value: clamping threshold.

    Returns:
        Softcapped logits (same shape, may be the same tensor).
    """
    return softcap_value * torch.tanh(logits / softcap_value)


def ce_softcap(
    logits: torch.Tensor,
    targets: torch.Tensor,
    softcap_value: float = 15.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss with logit softcap (pure-PyTorch fallback).

    Applies ``softcap`` then :func:`F.cross_entropy`.

    Args:
        logits: unnormalised logits ``(B, T, V)``.
        targets: target token IDs ``(B, T)``.
        softcap_value: softcap threshold.
        ignore_index: target value to ignore in the loss.

    Returns:
        Scalar loss.
    """
    logits = softcap(logits, softcap_value)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


if _HAS_TRITON:

    @triton.jit
    def _ce_softcap_fwd_kernel(
        logits_ptr,
        targets_ptr,
        loss_ptr,
        stride_logits,
        stride_targets,
        V: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Triton forward kernel for fused CE + softcap.

        Each program processes one row (one token position).
        """
        row = tl.program_id(0)
        logits_offset = logits_ptr + row * stride_logits
        target = tl.load(targets_ptr + row * stride_targets)

        if target == IGNORE:
            tl.store(loss_ptr + row, 0.0)
            return

        offs = tl.arange(0, BLOCK_V)
        logits_row = tl.load(logits_offset + offs, mask=offs < V, other=0.0)
        logits_row = SOFTCAP * tl.math.tanh(logits_row / SOFTCAP)

        c = logits_row - tl.max(logits_row, axis=0)
        logsumexp = tl.log(tl.sum(tl.exp(c), axis=0))
        loss_val = logsumexp - tl.load(logits_row + target)
        tl.store(loss_ptr + row, loss_val)

    def _triton_ce_softcap(logits, targets, softcap_value=15.0, ignore_index=-100):
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        n_rows = flat_logits.size(0)
        loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

        BLOCK_V = 512 if V <= 512 else 1024
        grid = (n_rows,)
        _ce_softcap_fwd_kernel[grid](
            flat_logits,
            flat_targets,
            loss,
            flat_logits.stride(0),
            flat_targets.stride(0),
            V=V,
            SOFTCAP=softcap_value,
            IGNORE=ignore_index,
            BLOCK_V=BLOCK_V,
        )
        return loss.mean()

    def fused_ce_softcap(
        logits: torch.Tensor,
        targets: torch.Tensor,
        softcap_value: float = 15.0,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Fused CE + softcap via Triton (fast path).

        Falls back to the pure-PyTorch version when triton is unavailable.
        """
        if logits.device.type != "cuda":
            return ce_softcap(logits, targets, softcap_value, ignore_index)
        return _triton_ce_softcap(logits, targets, softcap_value, ignore_index)

else:

    def fused_ce_softcap(
        logits: torch.Tensor,
        targets: torch.Tensor,
        softcap_value: float = 15.0,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Fused CE + softcap (pure-PyTorch fallback)."""
        return ce_softcap(logits, targets, softcap_value, ignore_index)
