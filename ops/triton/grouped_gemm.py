# ops/triton/grouped_gemm.py
"""Triton grouped-GEMM kernel (autotuned).

Production fast-path for MoE: when ``topk`` is small and the number
of active experts is large, the scatter-gather path in
:func:`models.moe.DeepSeekMoE.forward` is bandwidth-bound.  A
grouped-GEMM that processes *all* active experts in a single kernel
launch cuts kernel-launch overhead and reuses on-chip memory.

Kernel design
-------------
For each active expert ``e`` with ``M_e`` tokens and weight ``W_e``
of shape ``(K, N)``, we compute ``y_e = x_e @ W_e``.  The kernel
parallelizes over experts and over the output dimension ``N``:

    grid: (E, N / BLOCK_N)
    for each block:
        expert_e, n_offset = block indices
        M_e = offsets[e+1] - offsets[e]
        x_base = a[offsets[e]:offsets[e+1]]
        w_base = b[e]
        # Standard tile-based matmul with loop over K dimension

Fallback semantics
------------------
When ``triton`` is not available the wrapper raises
``NotImplementedError`` and the caller must fall back to the
per-expert scatter-gather loop.

Test surface
----------
A GPU-marked test in :mod:`tests.test_moe_gemm` verifies numerical
equivalence to the scatter-gather path within 1e-3 (BF16).
"""

from __future__ import annotations

from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def has_triton() -> bool:
    """Return True if Triton is importable and a CUDA device is available."""
    return _HAS_TRITON and torch.cuda.is_available()


# ── Autotune configs ──────────────────────────────────────────────────────


def _autotune_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
        for bm in [16, 32, 64]
        for bn in [32, 64, 128]
        for bk in [32, 64]
    ]


# ── Triton kernel ─────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.autotune(configs=_autotune_configs(), key=["K", "N"])
    @triton.jit
    def _grouped_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        offsets_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        E: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_be: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute grouped GEMM: ``c[e] = a[offsets[e]:offsets[e+1]] @ b[e]``.

        Grid: ``(E, N // BLOCK_N)``.

        Each block computes the output columns ``[n_off, n_off + BLOCK_N)``
        for expert ``e``, processing the K dimension in chunks of ``BLOCK_K``.
        """
        e_id = tl.program_id(0)
        n_off = tl.program_id(1) * BLOCK_N

        # Load expert offset → number of tokens for this expert.
        offs_e = tl.load(offsets_ptr + e_id).to(tl.int64)
        offs_e1 = tl.load(offsets_ptr + e_id + 1).to(tl.int64)
        m_size = offs_e1 - offs_e

        # Iterate over blocks of M tokens.
        for m_start in tl.static_range(0, m_size, BLOCK_M):
            m_off = m_start + tl.arange(0, BLOCK_M)

            # Accumulator for this tile of output.
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Loop over K dimension in chunks.
            for k_start in tl.static_range(0, K, BLOCK_K):
                k_off = k_start + tl.arange(0, BLOCK_K)

                # Load a tile of the input: a[offsets[e] + m_off, k_off]
                a_ptrs = (
                    a_ptr
                    + (offs_e + m_off[:, None]) * stride_am
                    + k_off[None, :] * stride_ak
                )
                a_tile = tl.load(
                    a_ptrs,
                    mask=(m_off[:, None] < offs_e + m_size)
                    & (k_off[None, :] < K),
                    other=0.0,
                )

                # Load a tile of the weight: b[e, k_off, n_off]
                b_ptrs = (
                    b_ptr
                    + e_id * stride_be
                    + k_off[:, None] * stride_bk
                    + (n_off + tl.arange(0, BLOCK_N))[None, :] * stride_bn
                )
                b_tile = tl.load(
                    b_ptrs,
                    mask=(k_off[:, None] < K)
                    & ((n_off + tl.arange(0, BLOCK_N))[None, :] < N),
                    other=0.0,
                )

                acc += tl.dot(a_tile.to(tl.float16), b_tile.to(tl.float16))

            # ── Write output ──────────────────────────────────────────
            out_ptrs = (
                c_ptr
                + (offs_e + m_off[:, None]) * K  # c is (M, N), use M as stride
                + (n_off + tl.arange(0, BLOCK_N))[None, :]
            )
            tl.store(
                out_ptrs,
                acc.to(a_ptr.dtype.element_ty),
                mask=(m_off[:, None] < offs_e + m_size)
                & ((n_off + tl.arange(0, BLOCK_N))[None, :] < N),
            )


# ── Python wrapper ────────────────────────────────────────────────────────


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute grouped GEMM: ``y[e] = a[offsets[e]:offsets[e+1]] @ b[e]``.

    Args:
        a:              ``(M, K)`` — tokens routed to this batch.
        b:              ``(E, K, N)`` — one weight matrix per expert.
        expert_offsets: ``(E+1,)`` int32 — cumulative token counts.

    Returns
    -------
    ``(M, N)`` result tensor.
    """
    if not has_triton():
        raise NotImplementedError("Triton not available; fall back to the scatter-gather path")

    M, K = a.shape
    E, _K, N = b.shape
    assert _K == K, f"K dim mismatch: {_K} vs {K}"
    assert expert_offsets.shape == (E + 1,), f"expected offsets ({E + 1},), got {expert_offsets.shape}"
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)

    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_N"]))
    _grouped_gemm_kernel[grid](
        a, b, c, expert_offsets,
        M, K, N, E,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
    )
    return c


__all__ = ["grouped_gemm", "has_triton"]
