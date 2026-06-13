# models/moe.py
"""DeepSeekMoE: Mixture-of-Experts.

Architecture:
  Input (T, 768)
    ├─ Gate: Linear(768→8) + bias → Sigmoid + bias → Top-2
    ├─ 8 Routed experts (SwiGLU, 768→2048→768): top-2 active per token
    ├─ 1 Shared expert (SwiGLU, 768→2048→768): always active
    └─ Output = Σ(weight_i × expert_i(x)) + shared_expert(x)

Per-layer params: 29,568,768
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert: y = SiLU(W1·x) ⊙ (W3·x); out = W2·y."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with aux-loss-free biased sigmoid routing.

    Specs:
      - 8 routed experts (top-2), 1 shared expert
      - moe_inter_dim = 2048
      - Aux-loss-free biased sigmoid gate
      - Scatter-gather dispatch
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]                            # 768
        self.n_routed_experts = config["n_routed_experts"]  # 8
        self.n_shared_experts = config["n_shared_experts"]  # 1
        self.n_activated_experts = config["n_activated_experts"]  # 2 (top-2)
        self.moe_inter_dim = config["moe_inter_dim"]        # 2048
        self.capacity_factor = config.get("expert_capacity_factor", 1.5)
        self.route_scale = config.get("route_scale", 1.0)

        # ── Gate: Linear(768→8) with bias (the only biased Linear in the model) ──
        self.gate = nn.Linear(self.dim, self.n_routed_experts, bias=True)
        # Gate bias is explicitly used for load balancing
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.006)

        # ── Routed experts ─────────────────────────────────────────────────
        self.experts = nn.ModuleList([
            SwiGLUExpert(self.dim, self.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])

        # ── Shared expert ──────────────────────────────────────────────────
        if self.n_shared_experts > 0:
            self.shared_expert = SwiGLUExpert(self.dim, self.moe_inter_dim)
        else:
            self.shared_expert = None

        # Cached routing state (for bias update)
        self._last_indices: torch.Tensor | None = None
        self._last_weights: torch.Tensor | None = None

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aux-loss-free biased sigmoid routing.

        Args:
            x: (T, dim) flattened tokens

        Returns:
            weights: (T, topk) routing weights (normalised, sum-to-1 per token)
            indices: (T, topk) global expert indices
        """
        # Raw logits: (T, n_routed_experts)
        logits = self.gate(x)
        # Sigmoid scores
        scores = torch.sigmoid(logits)
        # Top-k by score
        weights, indices = torch.topk(scores, k=self.n_activated_experts, dim=-1)
        # Normalise weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        weights = weights * self.route_scale
        return weights, indices

    def _dispatch_scatter_gather(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter-gather dispatch for routed experts.

        Iterates over active experts, gathers assigned tokens,
        computes expert forward, scatters back in FP32 for numerical stability.
        """
        T = x.size(0)
        topk = weights.size(1)
        n_experts = self.n_routed_experts
        device = x.device

        # Build flat assignment arrays
        flat_indices = indices.flatten()                   # (T*topk,)
        flat_weights = weights.flatten().unsqueeze(-1)     # (T*topk, 1)
        flat_token_ids = torch.arange(T, device=device).repeat_interleave(topk)  # (T*topk,)

        # Sort by expert index for efficient access
        sort_idx = flat_indices.argsort()
        flat_indices_sorted = flat_indices[sort_idx]
        flat_weights_sorted = flat_weights[sort_idx]
        flat_token_ids_sorted = flat_token_ids[sort_idx]

        # Find unique experts and their segment sizes
        unique_experts, counts = torch.unique_consecutive(flat_indices_sorted, return_counts=True)
        starts = torch.zeros_like(counts)
        starts[1:] = counts.cumsum(0)[:-1]

        # Output buffer (accumulate in FP32 for numerical stability)
        y = torch.zeros(T, self.dim, dtype=torch.float32, device=device)

        for i in range(unique_experts.size(0)):
            expert_idx = unique_experts[i].item()
            if expert_idx >= n_experts:
                continue
            start = starts[i].item()
            count = counts[i].item()
            if count == 0:
                continue
            token_ids = flat_token_ids_sorted[start: start + count]
            w = flat_weights_sorted[start: start + count]
            expert_out = self.experts[expert_idx](x[token_ids])
            y.index_add_(0, token_ids, (expert_out * w).to(y.dtype))

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (T, dim) or (B, T, dim) — token representations

        Returns:
            Same shape as x.
        """
        shape = x.shape
        flat = x.view(-1, self.dim)
        T = flat.size(0)

        # ── Routing ────────────────────────────────────────────────────────
        weights, indices = self._route(flat)
        # Cache for bias update
        self._last_indices = indices.detach()
        self._last_weights = weights.detach()

        # ── Routed expert output (top-2 per token) ─────────────────────────
        y_routed = self._dispatch_scatter_gather(flat, weights, indices)

        # ── Shared expert output ──────────────────────────────────────────
        if self.shared_expert is not None:
            y_shared = self.shared_expert(flat)
        else:
            y_shared = 0.0

        return (y_routed + y_shared).view(shape)

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """Update gate bias for load balancing (aux-loss-free).

        Called every bias_update_every steps. Over-loaded experts
        have their bias decreased; under-loaded have it increased.
        """
        if self._last_indices is None:
            return
        counts = torch.bincount(
            self._last_indices.flatten(),
            minlength=self.n_routed_experts,
        ).float()
        avg = counts.mean()
        over = counts > avg * 1.10
        under = counts < avg * 0.90
        with torch.no_grad():
            self.gate.bias[over] -= speed
            self.gate.bias[under] += speed

    def get_load_balance_loss(self) -> torch.Tensor:
        """Auxiliary load balance loss (safety floor, alpha=1e-4)."""
        if self._last_indices is None or self._last_weights is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        T = self._last_weights.size(0)
        n_experts = self.n_routed_experts
        topk = self._last_weights.size(1)
        counts = torch.bincount(
            self._last_indices.flatten(), minlength=n_experts
        ).float()
        f = counts / (counts.sum() + 1e-10)
        # Mean routing probability per expert
        one_hot = F.one_hot(self._last_indices.flatten(), num_classes=n_experts).float()
        P = (one_hot * self._last_weights.flatten().unsqueeze(-1)).view(T, topk, n_experts).sum(dim=1).mean(dim=0)
        return (f * P).sum() * n_experts
