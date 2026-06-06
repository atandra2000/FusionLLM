# kernels/flash_attn.py
"""Flash-Attention 3 dispatch wrapper.

Provides a single function :func:`flash_attention` that:
1. Tries to use ``flash_attn.flash_attn_func`` (Flash-Attention 3) when
   the package is installed and ``use_fa3=True``.
2. Falls back to ``F.scaled_dot_product_attention`` (PyTorch native
   sdpa) on CPU or when FA3 is not requested.

Also provides :func:`long_short_window_mask` for the Gemma 2 / DeepSeek
style long-short window pattern where every ``period``-th block gets
global attention and the rest use a sliding window.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_HAS_FLASH_ATTN: bool = False
try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except Exception:
    pass


def has_flash_attn() -> bool:
    """Return True if the ``flash_attn`` package is installed and CUDA is available."""
    return _HAS_FLASH_ATTN and torch.cuda.is_available()


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    use_fa3: bool = True,
) -> torch.Tensor:
    """Dispatch to FA3 or pytorch SDPA.

    Args:
        query:  ``(bsz, n_heads, seqlen, head_dim)``.
        key:    ``(bsz, n_heads, seqlen, head_dim)``.
        value:  ``(bsz, n_heads, seqlen, head_dim)``.
        attn_mask: Optional additive mask ``(bsz, 1, seqlen, seqlen)``
                   or ``(1, 1, seqlen, seqlen)``.  Ignored when FA3 is
                   active (FA3 uses causal masking internally).
        scale:  Softmax scaling factor (default ``head_dim ** -0.5``).
        use_fa3: Whether to attempt the FA3 kernel.

    Returns
    -------
    ``(bsz, n_heads, seqlen, head_dim)`` — attention output.
    """
    if use_fa3 and has_flash_attn():
        if attn_mask is not None:
            dtype = query.dtype
            q = query.transpose(1, 2).to(torch.bfloat16)
            k = key.transpose(1, 2).to(torch.bfloat16)
            v = value.transpose(1, 2).to(torch.bfloat16)
            out = flash_attn_func(q, k, v, softmax_scale=scale or (query.size(-1) ** -0.5))
            return out.transpose(1, 2).to(dtype)
        else:
            dtype = query.dtype
            q = query.transpose(1, 2).to(torch.bfloat16)
            k = key.transpose(1, 2).to(torch.bfloat16)
            v = value.transpose(1, 2).to(torch.bfloat16)
            out = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=scale or (query.size(-1) ** -0.5),
                causal=True,
            )
            return out.transpose(1, 2).to(dtype)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, scale=scale)


def long_short_window_mask(
    n_layers: int,
    seqlen: int,
    end_pos: int,
    device: torch.device,
    window: int = 2048,
    period: int = 5,
) -> list[torch.Tensor | None]:
    """Build attention masks for a long-short window schedule.

    In a (period - 1):1 pattern, every ``period``-th layer gets
    **global** attention (no sliding window), and the other layers
    get **local** (sliding) attention over ``window`` tokens.

    Args:
        n_layers:  Total number of attention layers.
        seqlen:    Query sequence length.
        end_pos:   Total KV length (``start_pos + seqlen``).
        device:    Target device.
        window:    Sliding window radius (number of past tokens).
        period:    Global-attention period (e.g. 5 → every 5th layer).

    Returns
    -------
    A list of ``n_layers`` masks (or ``None`` for global-attention
    layers), each shaped ``(1, 1, seqlen, end_pos)``.
    """
    masks: list[torch.Tensor | None] = []
    for layer_idx in range(n_layers):
        if (layer_idx + 1) % period == 0:
            masks.append(None)
        else:
            i = torch.arange(end_pos, device=device)
            j = torch.arange(end_pos, device=device)
            causal = j[None, :] <= i[:, None]
            local = j[None, :] >= i[:, None] - (window - 1)
            mask = torch.where(causal & local, 0.0, float("-inf"))
            mask = mask[-seqlen:, :].unsqueeze(0).unsqueeze(0)
            masks.append(mask)
    return masks


__all__ = ["flash_attention", "long_short_window_mask", "has_flash_attn"]
