# models/moe.py
"""DeepSeekMoE: Mixture-of-Experts (aux-loss-free biased sigmoid routing)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """SwiGLU expert."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with aux-loss-free biased sigmoid routing."""

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.n_activated_experts = config["n_activated_experts"]
        self.moe_inter_dim = config["moe_inter_dim"]
        self.route_scale = config.get("route_scale", 1.0)

        self.gate = nn.Linear(self.dim, self.n_routed_experts, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.006)
        self.experts = nn.ModuleList([SwiGLUExpert(self.dim, self.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_expert = SwiGLUExpert(self.dim, self.moe_inter_dim) if self.n_shared_experts > 0 else None
        self._last_indices: torch.Tensor | None = None
        self._last_weights: torch.Tensor | None = None

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Biased sigmoid routing (top-k)."""
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        weights, indices = torch.topk(scores, k=self.n_activated_experts, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10) * self.route_scale
        return weights, indices

    def _dispatch_scatter_gather(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Scatter-gather dispatch."""
        T, topk, n_experts, device = x.size(0), weights.size(1), self.n_routed_experts, x.device
        flat_indices = indices.flatten()
        flat_weights = weights.flatten().unsqueeze(-1)
        flat_token_ids = torch.arange(T, device=device).repeat_interleave(topk)
        sort_idx = flat_indices.argsort()
        flat_indices_sorted = flat_indices[sort_idx]
        flat_weights_sorted = flat_weights[sort_idx]
        flat_token_ids_sorted = flat_token_ids[sort_idx]
        unique_experts, counts = torch.unique_consecutive(flat_indices_sorted, return_counts=True)
        starts = torch.zeros_like(counts)
        starts[1:] = counts.cumsum(0)[:-1]
        y = torch.zeros(T, self.dim, dtype=torch.float32, device=device)
        for i in range(unique_experts.size(0)):
            expert_idx = unique_experts[i].item()
            if expert_idx >= n_experts:
                continue
            start, count = starts[i].item(), counts[i].item()
            if count == 0:
                continue
            token_ids = flat_token_ids_sorted[start: start + count]
            w = flat_weights_sorted[start: start + count]
            expert_out = self.experts[expert_idx](x[token_ids])
            y.index_add_(0, token_ids, (expert_out * w).to(y.dtype))
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.view(-1, self.dim)
        weights, indices = self._route(flat)
        self._last_indices = indices.detach()
        self._last_weights = weights.detach()
        y_routed = self._dispatch_scatter_gather(flat, weights, indices)
        y_shared = self.shared_expert(flat) if self.shared_expert else 0.0
        return (y_routed + y_shared).view(shape)

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """Update gate bias for load balancing."""
        if self._last_indices is None:
            return
        counts = torch.bincount(self._last_indices.flatten(), minlength=self.n_routed_experts).float()
        avg = counts.mean()
        over, under = counts > avg * 1.10, counts < avg * 0.90
        with torch.no_grad():
            self.gate.bias[over] -= speed
            self.gate.bias[under] += speed

    def get_load_balance_loss(self) -> torch.Tensor:
        """Auxiliary load balance loss."""
        if self._last_indices is None or self._last_weights is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        T, n_experts, topk = self._last_weights.size(0), self.n_routed_experts, self._last_weights.size(1)
        counts = torch.bincount(self._last_indices.flatten(), minlength=n_experts).float()
        f = counts / (counts.sum() + 1e-10)
        one_hot = F.one_hot(self._last_indices.flatten(), num_classes=n_experts).float()
        P = (one_hot * self._last_weights.flatten().unsqueeze(-1)).view(T, topk, n_experts).sum(dim=1).mean(dim=0)
        return (f * P).sum() * n_experts
