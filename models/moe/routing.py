# models/moe/routing.py
"""Auxiliary-loss-free routing gate and segment computation.

DeepSeek-V3 style routing: sigmoid-biased scores with optional
group-limited (node-limited) routing.  The bias is updated after each
optimizer step using cached token counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_routing_segments(
    T: int,
    topk: int,
    flat: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    n_local_experts: int,
    experts_start: int,
    n_routed_experts: int,
    capacity_factor: float,
    dropout_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared sort/segment/capacity logic for scatter-gather routing.

    Returns:
        flat_token_ids_sorted, flat_weights_sorted, expert_start, expert_size, active_indices
    """
    total_assign = T * topk
    flat_token_ids = torch.arange(T, device=flat.device).repeat_interleave(topk)
    flat_expert_global = indices.flatten()
    flat_weights = weights.flatten()

    sort_idx = flat_expert_global.argsort()
    flat_expert_global_sorted = flat_expert_global[sort_idx]
    flat_token_ids_sorted = flat_token_ids[sort_idx]
    flat_weights_sorted = flat_weights[sort_idx]

    unique_experts, segment_sizes = torch.unique_consecutive(
        flat_expert_global_sorted, return_counts=True
    )
    segment_starts = torch.cat(
        [torch.tensor([0], device=flat.device), segment_sizes.cumsum(0)[:-1]]
    )

    expert_start = torch.full((n_local_experts,), -1, dtype=torch.long, device=flat.device)
    expert_size = torch.zeros(n_local_experts, dtype=torch.long, device=flat.device)

    valid_mask = (unique_experts >= experts_start) & (
        unique_experts < experts_start + n_local_experts
    )
    local_expert_ids = unique_experts[valid_mask] - experts_start
    expert_start[local_expert_ids] = segment_starts[valid_mask]
    expert_size[local_expert_ids] = segment_sizes[valid_mask]

    expected_assign = (total_assign + n_routed_experts - 1) // n_routed_experts
    capacity = int(capacity_factor * expected_assign)
    expert_size = torch.minimum(
        expert_size, torch.tensor(capacity, dtype=torch.long, device=flat.device)
    )

    active_mask = dropout_mask & (expert_start >= 0) & (expert_size > 0)
    active_indices = torch.where(active_mask)[0]

    return flat_token_ids_sorted, flat_weights_sorted, expert_start, expert_size, active_indices


class AuxLossFreeGate(nn.Module):
    """
    Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3).

    Routing decision
    ----------------
    Each token is assigned to the top-k experts by a biased score:

        biased_score_e = sigmoid(x @ W_e^T) + bias_e

    The bias is NOT used when computing the final routing weights — only the raw sigmoid scores are load
    normalised and used as weights.  This separates load balancing (via bias) from the gradient
    signal (via raw scores).

    Group-limited routing
    ---------------------
    When n_groups > 1 the experts are divided into n_groups equal groups.
    Only topk_groups groups are selected per token (node-limited routing).
    Within each selected group the top-`group_topk` biased scores are summed
    to produce a group score; the top groups by that score are activated.

    Bias update
    -----------
    After each optimiser step the caller should invoke update_bias() with the per-expert token counts from the
    last forward pass.  Experts that are over-loaded (count > avg * (1 + upper_threshold)) have their bias
    decreased; under-loaded experts have their bias increased.  The bias is stored as a plain buffer
    (not a Parameter) so it does not appear in optimiser state dicts.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.topk = config["n_activated_experts"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_groups = config.get("n_expert_groups", 1)
        self.topk_groups = config.get("n_limited_groups", 1)
        self.route_scale = config.get("route_scale", 1.0)
        self.group_topk = config.get("group_topk", 2)
        self.bias_upper = config.get("bias_upper_threshold", 0.10)
        self.bias_lower = config.get("bias_lower_threshold", 0.10)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
        nn.init.normal_(self.weight, std=0.006)
        self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._last_router_logits: torch.Tensor | None = None

    def get_z_loss(self) -> torch.Tensor:
        return (
            self._zero.to(self.weight.device)
            if self._last_router_logits is None
            else torch.logsumexp(self._last_router_logits, dim=-1).pow(2).mean()
        )

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
        counts = counts.float()
        avg = counts.mean()
        over = counts > avg * (1.0 + self.bias_upper)
        under = counts < avg * (1.0 - self.bias_lower)
        self.bias[over] -= speed
        self.bias[under] += speed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, dim) — flattened token representations

        Returns:
            weights: (T, topk) — normalised routing weights (sum-to-1 per token,
                                  then scaled by route_scale)
            indices: (T, topk) — global expert indices
        """
        T = x.size(0)

        # Pre-sigmoid logits (cached for z-loss in the trainer)
        router_logits = F.linear(x, self.weight)  # (T, E)
        # Clamp logits to ±30 to avoid logsumexp overflow in z-loss
        router_logits = router_logits.clamp(-30, 30)
        self._last_router_logits = router_logits.detach()

        # Raw sigmoid scores — used for final weight computation
        scores = router_logits.sigmoid()  # (T, E)

        # Biased scores — used for routing decision only
        biased = scores + self.bias.to(scores.dtype)  # (T, E)

        if self.n_groups > 1:
            experts_per_group = self.n_routed_experts // self.n_groups
            # (T, n_groups, experts_per_group)
            biased_grouped = biased.view(T, self.n_groups, experts_per_group)
            # Group score = sum of top-group_topk biased scores within each group
            group_scores = biased_grouped.topk(self.group_topk, dim=-1)[0].sum(dim=-1)
            # Select topk_groups groups per token
            top_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # (T, topk_groups)
            # Mask out non-selected groups
            group_mask = torch.ones(T, self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, top_groups, False)
            biased = biased_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(
                1
            )  # (T, E)

        # Select top-k experts by biased score
        indices = biased.topk(self.topk, dim=-1)[1]  # (T, topk)

        # Routing weights from raw (unbiased) scores at the selected positions
        weights = scores.gather(1, indices)  # (T, topk)

        # Normalise so weights sum to 1 per token, then apply route_scale
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        weights = (weights * self.route_scale).to(x.dtype)

        return weights, indices
