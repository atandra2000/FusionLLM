# kernels/linear_relu2.py
"""Fused Linear + ReLU² kernel.

The operation is::

    out = relu(x @ W.T) ** 2

where ``relu(z) = max(0, z)`` and the square is element-wise.

Fusing the linear and the activation avoids materialising the
intermediate linear output on GPU, saving memory and bandwidth.

This file provides:

* :func:`linear_relu2` — pure-PyTorch fallback (always available).
* :func:`fused_linear_relu2` — Triton kernel when CUDA is present.

Reference
---------
Keller Jordan, "modded-nanogpt speedrun", record #59 (April 2026).
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


def linear_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear + ReLU²  (pure-PyTorch fallback).

    ``out = relu(x @ W.T + bias) ** 2``

    Args:
        x: input ``(..., in_features)``.
        weight: ``(out_features, in_features)``.
        bias: optional bias ``(out_features,)``.

    Returns:
        Output ``(..., out_features)``.
    """
    out = F.linear(x, weight, bias)
    return torch.relu(out).pow(2)


if _HAS_TRITON:

    @triton.jit
    def _linear_relu2_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_outm,
        stride_outn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Triton forward kernel for fused Linear + ReLU².

        Each program processes a ``BLOCK_M × BLOCK_N`` tile of the output.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
            w_block = tl.load(w_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(x_block, w_block, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_n
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        acc = tl.where(acc < 0, 0.0, acc)
        acc = acc * acc

        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def fused_linear_relu2(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused Linear + ReLU² via Triton.

        Falls back to the pure-PyTorch version when triton or CUDA is
        unavailable.
        """
        if x.device.type != "cuda":
            return linear_relu2(x, weight, bias)

        M, K = x.shape
        N, K_w = weight.shape
        assert K == K_w, f"x: {x.shape}, weight: {weight.shape}"

        out = torch.empty(M, N, device=x.device, dtype=x.dtype)
        has_bias = bias is not None

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        _linear_relu2_fwd_kernel[grid](
            x,
            weight,
            bias if has_bias else weight,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HAS_BIAS=has_bias,
        )
        return out

else:

    def fused_linear_relu2(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused Linear + ReLU² (pure-PyTorch fallback)."""
        return linear_relu2(x, weight, bias)
