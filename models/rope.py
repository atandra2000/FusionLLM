# models/rope.py
"""Shared RoPE (Rotary Position Embedding) with optional YaRN scaling.

This module is the single source of truth for rotary embedding
construction and application across the codebase (MLA, GDN, etc.).

Key properties
---------------
* **Device-aware grow strategy** — the `freqs_cis` buffer is rebuilt
  on demand when the requested sequence length exceeds the cached
  length, *and* the device changes (e.g. after `.to('cuda')`).
* **YaRN non-uniform scaling** — when ``rope_factor > 1`` the
  per-dimension inverse frequencies are scaled to compress the
  high-frequency end (a la DeepSeek-V3 YaRN).
* **Numerical equivalence to the legacy mla.py implementation** —
  tests assert 1e-5 max abs error against a frozen reference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _yarn_freq_scaling(inv_freq: torch.Tensor, rope_factor: float, dim: int) -> torch.Tensor:
    """Apply non-uniform YaRN frequency scaling.

    The standard YaRN formula: each dimension's inverse frequency is
    scaled by ``rope_factor ** (2i / dim)``.  When ``rope_factor <= 1``
    this is a no-op (we still return a fresh tensor to avoid in-place
    surprises).
    """
    if rope_factor <= 1.0:
        return inv_freq.clone()
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=inv_freq.device)
    scale = rope_factor ** (i / dim)
    return inv_freq / scale


def apply_rope(
    x: torch.Tensor,
    start_pos: int,
    seqlen: int,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to a query/key tensor.

    Args:
        x:          ``(..., seqlen, head_dim)`` with ``head_dim``
                    even.  Will be cast to complex internally.
        start_pos:  starting position offset (for KV-cache).
        seqlen:     length of the sequence to apply.
        freqs_cis:  complex buffer ``(max_seq, head_dim/2)``.

    Returns
    -------
    ``x`` shape-preserving, with the rope applied in its input dtype.
    """
    dtype = x.dtype
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[start_pos : start_pos + seqlen].view(1, seqlen, 1, -1)
    return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)


class RotaryEmbedding(nn.Module):
    """Per-layer rotary embedding table with YaRN scaling and grow-on-demand.

    Args:
        head_dim:    rotary head dim (must be even).  Typically
                     ``qk_rope_head_dim`` of the attention block.
        rope_theta:  base for the geometric progression (default 10 000).
        rope_factor: YaRN scaling factor; 1.0 = no scaling.
        max_seq_len: upper bound on sequence length; preallocates up
                     to this on the *first* request to avoid repeated
                     rebuilds during normal training.

    Buffers
    -------
    ``freqs_cis`` — complex64 table of shape
    ``(current_seq_len, head_dim/2)``.  Persistent=False so it
    doesn't bloat checkpoints.
    """

    freqs_cis: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.max_seq_len = max_seq_len
        self._cached_seq_len: int = 0
        self.register_buffer(
            "freqs_cis",
            torch.empty(0, head_dim // 2, dtype=torch.complex64),
            persistent=False,
        )

    def extend_to(self, end_pos: int, device: torch.device) -> None:
        """Ensure the table covers at least ``end_pos`` positions on ``device``.

        The table grows geometrically (``max(needed, cached*2, 64)``)
        up to ``max_seq_len`` to avoid repeated reallocations during
        long-context training.
        """
        fc = self.freqs_cis
        if (
            end_pos <= self._cached_seq_len
            and fc.device == device
            and fc.numel() > 0
        ):
            return
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"RotaryEmbedding: end_pos {end_pos} > max_seq_len {self.max_seq_len}"
            )
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )
        inv_freq = _yarn_freq_scaling(inv_freq, self.rope_factor, self.head_dim)
        grow_to = min(max(end_pos, self._cached_seq_len * 2, 64), self.max_seq_len)
        t = torch.arange(grow_to, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._cached_seq_len = grow_to

    def forward(
        self, x: torch.Tensor, start_pos: int = 0, seqlen: int | None = None
    ) -> torch.Tensor:
        """Apply RoPE to ``x`` (convenience wrapper).

        ``x`` is expected to have shape ``(..., seqlen, head_dim)`` —
        the second-to-last dim is the sequence length.  We pass it
        through to :func:`apply_rope` which broadcasts the
        ``(1, seqlen, 1, head_dim/2)`` frequency slice.
        """
        if seqlen is None:
            seqlen = x.size(-2)
        self.extend_to(start_pos + seqlen, x.device)
        return apply_rope(x, start_pos, seqlen, self.freqs_cis)
