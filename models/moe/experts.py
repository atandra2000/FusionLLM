# models/moe/experts.py
"""Single expert FFN module.

Supports SwiGLU and ReLU² activations with optional tensor-parallel
weight splitting.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert FFN.

    Activation is configurable per-instance:
    * ``"swiglu"`` (default) — ``W2(SiLU(W1(x)) * W3(x))`` (3 weight
      matrices, matches Llama 3 / Qwen 2.5 / Gemma 2 / OLMo 2).
    * ``"relu2"`` (legacy) — ``W2(relu(W1(x)) ** 2)`` (2 weight
      matrices, the original MoE expert).  Kept for the legacy
      ReLU² config; the canonical profile uses SwiGLU.

    All linears are constructed with ``bias=False`` to match the
    field (Qwen 3, Nemotron-H, OLMo 2).
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.activation = activation

        if tp_size > 1:
            assert inter_dim % tp_size == 0, (
                f"inter_dim ({inter_dim}) must be divisible by tp_size ({tp_size})"
            )
            tp_inter_dim = inter_dim // tp_size
        else:
            tp_inter_dim = inter_dim

        if activation == "swiglu":
            self.w1 = nn.Linear(dim, tp_inter_dim, bias=False)
            self.w2 = nn.Linear(tp_inter_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, tp_inter_dim, bias=False)
        elif activation == "relu2":
            self.w1 = nn.Linear(dim, tp_inter_dim, bias=False)
            self.w2 = nn.Linear(tp_inter_dim, dim, bias=False)
            self.w3 = None
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            h = F.silu(self.w1(x)) * self.w3(x)
        else:  # relu2
            h = torch.relu(self.w1(x)).square()
        y_local = self.w2(h)
        if self.tp_size > 1:
            dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
        return y_local


def expert_forward_single(
    x_group: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor | None,
    activation: str,
) -> torch.Tensor:
    """Single expert forward pass using raw weight tensors (SwiGLU or ReLU²)."""
    h1 = F.linear(x_group, w1)
    if activation == "swiglu":
        h = F.silu(h1) * F.linear(x_group, w3)
    else:
        h = torch.relu(h1).square()
    return F.linear(h, w2)
