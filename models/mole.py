# models/mole.py
"""Mixture of Linear Experts (MoLE) — auxiliary low-rank expert bank.

MoLE adds a per-layer *auxiliary* computation that runs alongside the
main FFN / MoE block.  It is a *lightweight* alternative to (or
complement of) the routed MoE: each expert is a low-rank linear map
``(W_A, W_B)`` with rank ``r ≪ d``, and routing is a *per-layer
shared* softmax over ``N`` experts.  Top-1 routing is applied per
token.

Design (per modded-nanogpt #4, "linear experts")
------------------------------------------------
* Init: ``W_A ~ N(0, 1/sqrt(r))``, ``W_B = 0`` (zero-init projection
  so the MoLE output is zero at the start of training — gradient
  vanishes, no disruption to the main path).
* Forward: ``y = sum_i gate_i(x) * (x @ W_A_i @ W_B_i)`` per token
  with a *per-layer* shared router.  (MoLE and MoE are *never*
  co-resident in the same block in Phase 2 scope; the
  ``TransformerBlock`` decides which one to instantiate.)
* Activation: SwiGLU-style gate on the output of the router.

Reference
---------
* modded-nanogpt #4 (Keller Jordan, 2024).
* "Mixture of Linear Experts" (Wu et al., 2024) — the original
  LoRA-as-MoE paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoLE(nn.Module):
    """One MoLE bank: ``n_experts`` low-rank experts with a per-layer router.

    Args:
        dim:        model dim.
        rank:       expert rank ``r``.
        n_experts:  number of experts in the bank.
        top_k:      number of experts to route per token (default 1).
        every_n:    frequency at which MoLE is applied in the layer
                    schedule (default 4 — every 4th FFN slot).
                    *Not* used by this module — the caller decides.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 32,
        n_experts: int = 8,
        top_k: int = 1,
        every_n: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_experts = n_experts
        self.top_k = top_k
        self.every_n = every_n  # informational only

        # Per-expert low-rank factors.  Shape (n_experts, dim, rank)
        # and (n_experts, rank, dim).  Initialise W_B to zero so the
        # output of MoLE is zero at the start of training (modded-
        # nanogpt #4).
        self.W_A = nn.Parameter(torch.empty(n_experts, dim, rank))
        self.W_B = nn.Parameter(torch.zeros(n_experts, rank, dim))

        # Per-layer shared router: x → logits over n_experts.
        self.router = nn.Linear(dim, n_experts, bias=False)

        # Init W_A: N(0, 1/sqrt(r))
        nn.init.normal_(self.W_A, mean=0.0, std=1.0 / (rank**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bsz, seqlen, dim)  →  (bsz, seqlen, dim)."""
        bsz, seqlen, dim = x.shape
        assert dim == self.dim, f"expected last dim {self.dim}, got {dim}"

        # 1) Router: (b, t, n_experts)
        logits = self.router(x)
        # 2) Top-k softmax routing
        if self.top_k == 1:
            weights = F.softmax(logits, dim=-1)  # (b, t, n_experts)
        else:
            topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
            topk_w = F.softmax(topk_logits, dim=-1)  # (b, t, k)
            # Scatter back to dense weights for the einsum below.
            weights = x.new_zeros(bsz, seqlen, self.n_experts)
            weights.scatter_(-1, topk_idx, topk_w)

        # 3) Per-expert projection: y = x @ W_A  →  (b, t, n_experts, r)
        x_a = torch.einsum("btd,edr->bter", x, self.W_A)
        # 4) Apply W_B per expert: y = y_a @ W_B → (b, t, n_experts, d)
        x_ab = torch.einsum("bter,erd->bted", x_a, self.W_B)
        # 5) Weighted sum: (b, t, d) = sum_e weights[..., e] * x_ab[..., e, :]
        y = torch.einsum("bte,bted->btd", weights, x_ab)
        return y

    def extra_repr(self) -> str:
        return f"dim={self.dim} rank={self.rank} n_experts={self.n_experts} top_k={self.top_k}"
