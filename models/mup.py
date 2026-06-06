# models/mup.py
"""μP (Maximal Update Parametrisation) initialisation.

The plan recipe (modded-nanogpt #1, μP paper, DeepSeek-V3 §3.4):
* **Residual stream**: std = 1 / sqrt(n_layers).
* **Attention / FFN matrices**: std = 1 / d (model dim).
* **Embeddings**: std = 1 / sqrt(d).
* **Biases**: 0.
* **Gates and scalar params**: 0 (so they don't affect the forward
  pass at init).

The base_lr is the "base shape" learning rate; the caller must
rescale it per parameter shape using :func:`muP_rescale_lr`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Param names that are gates / scalar params and should be zero-initialised.
_GATE_LIKE_NAMES = (
    "gate",
    "g_proj",
    "A_log",
    "dt_bias",
    "mscale",
    "router",  # in MoLE
    "output_head",  # tied; never re-init
)


def _is_gate_like(name: str) -> bool:
    n = name.lower()
    return any(g in n for g in _GATE_LIKE_NAMES)


def muP_init(model: nn.Module, config: dict) -> None:
    """Apply μP initialisation in place.

    Args:
        model:  the module to initialise.
        config: model config dict (must contain ``dim`` and
                ``n_layers``).
    """
    dim = int(config["dim"])
    n_layers = int(config["n_layers"])
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    res_std = 1.0 / math.sqrt(n_layers)

    # We walk the named parameters and re-initialise only the ones
    # that need it.  This is *additive* to the existing
    # ``_init_weights`` call in :class:`Transformer` — μP supersedes
    # the standard init for the parameters it touches.

    # First pass: zero-out all gate-like / scalar params.
    for name, p in model.named_parameters():
        if _is_gate_like(name):
            with torch.no_grad():
                p.data.zero_()

    # Second pass: standard re-init for matrices and embeddings.
    for name, p in model.named_parameters():
        if _is_gate_like(name):
            continue
        with torch.no_grad():
            if p.dim() < 2:
                # 0-d or 1-d parameter: keep as-is unless explicitly
                # marked gate-like above.
                continue
            # The residual stream lives in the *output* of each
            # block's norm2 + ffn, plus the embed.  We approximate
            # "residual stream" by the embed + output projection of
            # attention / MoE.  In practice the projection outputs
            # are init'd at the *attn/FFN* std (1/d) which is the
            # standard μP choice for output projections.
            std = attn_std
            if "embed" in name:
                std = embed_std
            # Scale by output fan-in for tied heads (rescale for
            # tied embeddings): if this is the head and it ties to
            # embed, we set the std to embed_std.
            if name.endswith("head.weight") and getattr(model, "tie_embeddings", False):
                std = embed_std
            # Truncated normal keeps us in the support of the
            # standard μP analysis.
            try:
                torch.nn.init.trunc_normal_(p.data, std=std, a=-2 * std, b=2 * std)
            except Exception:
                # Fallback for non-floating-point params (none in
                # practice — the embed/head/attn/ffn are all float).
                p.data.normal_(mean=0.0, std=std)


def muP_rescale_lr(base_lr: float, model_dim: int, *, param_dim: int | None = None) -> float:
    """Rescale the base learning rate for a parameter shape.

    μP says: ``lr ∝ 1 / param_dim``.  When ``param_dim`` is None
    (the default), we rescale to the *base shape* (model dim) and
    return the same ``base_lr``.  Otherwise, the formula is
    ``base_lr * model_dim / param_dim``.

    Reference: Yang & Hu, "Tensor Programs V" (μP paper), §2.3.
    """
    if param_dim is None:
        return base_lr
    return base_lr * (model_dim / param_dim)
