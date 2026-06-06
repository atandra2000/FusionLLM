"""Unit tests for `models/moe.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `AuxLossFreeGate` — group-limited routing, weight normalisation,
    bias update, z-loss.
  * `Expert` — both SwiGLU and ReLU² activations.
  * `DeepSeekMoE` — shared-expert invariant: replicated on every rank.
  * `compute_routing_segments` — segment math, capacity clipping.
"""

from __future__ import annotations

import pytest
import torch

from models.moe import (
    AuxLossFreeGate,
    DeepSeekMoE,
    Expert,
    compute_routing_segments,
)


# ── AuxLossFreeGate ────────────────────────────────────────────────────────
class TestAuxLossFreeGate:
    def _gate(self, n_routed: int = 4, topk: int = 2, n_groups: int = 1, **kw):
        cfg = dict(
            dim=8,
            n_routed_experts=n_routed,
            n_activated_experts=topk,
            n_expert_groups=n_groups,
            n_limited_groups=1,
            group_topk=1,
            route_scale=1.0,
            bias_upper_threshold=0.1,
            bias_lower_threshold=0.1,
        )
        cfg.update(kw)
        return AuxLossFreeGate(cfg)

    def test_weights_normalised_to_one(self):
        g = self._gate()
        x = torch.randn(16, 8)
        w, idx = g(x)
        assert torch.allclose(w.sum(-1), torch.ones(16), atol=1e-5)

    def test_indices_in_valid_range(self):
        g = self._gate(n_routed=4, topk=2)
        x = torch.randn(8, 8)
        _, idx = g(x)
        assert idx.min() >= 0 and idx.max() < 4

    def test_group_limited_routing(self):
        # 4 experts, 2 groups, topk_groups=1, topk=2 → each token
        # routes only within one group
        g = self._gate(n_routed=4, topk=2, n_groups=2, n_limited_groups=1, group_topk=1)
        x = torch.randn(32, 8)
        _, idx = g(x)
        # Group 0 = experts [0, 1], group 1 = experts [2, 3]
        for row in idx:
            assert all(0 <= e < 2 for e in row) or all(2 <= e < 4 for e in row)

    def test_bias_increases_for_under_loaded(self):
        g = self._gate(n_routed=4, topk=1)
        # Make expert 0 over-loaded and the rest under-loaded
        counts = torch.tensor([100.0, 1.0, 1.0, 1.0])
        before = g.bias.clone()
        g.update_bias(counts, speed=0.1)
        # over-loaded → bias decreases; under-loaded → bias increases
        assert g.bias[0] < before[0]
        for i in (1, 2, 3):
            assert g.bias[i] > before[i]

    def test_z_loss_zero_before_forward(self):
        g = self._gate()
        z = g.get_z_loss()
        assert z.item() == 0.0


# ── Expert ────────────────────────────────────────────────────────────────
class TestExpert:
    def test_swiglu_has_three_linears(self):
        e = Expert(dim=8, inter_dim=16, activation="swiglu")
        assert e.w1.out_features == 16 and e.w2.in_features == 16
        assert e.w3 is not None

    def test_relu2_has_two_linears(self):
        e = Expert(dim=8, inter_dim=16, activation="relu2")
        assert e.w1.out_features == 16 and e.w2.in_features == 16
        assert e.w3 is None

    def test_swiglu_forward_shape(self):
        e = Expert(dim=8, inter_dim=16, activation="swiglu")
        x = torch.randn(4, 8)
        y = e(x)
        assert y.shape == (4, 8)

    def test_relu2_forward_shape(self):
        e = Expert(dim=8, inter_dim=16, activation="relu2")
        x = torch.randn(4, 8)
        y = e(x)
        assert y.shape == (4, 8)


# ── compute_routing_segments ───────────────────────────────────────────────
class TestRoutingSegments:
    def test_segment_math(self):
        T, topk = 4, 2
        flat = torch.zeros(T, 4)
        # Token 0 → expert 0 (twice — topk=2)
        # Token 1 → expert 1
        # Token 2 → expert 0
        # Token 3 → expert 1
        indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        weights = torch.full((T, topk), 0.5)
        n_local, experts_start, n_routed = 2, 0, 2
        capacity = 100.0
        dropout_mask = torch.ones(n_local, dtype=torch.bool)
        flat_ids, flat_w, start, size, active = compute_routing_segments(
            T,
            topk,
            flat,
            indices,
            weights,
            n_local,
            experts_start,
            n_routed,
            capacity,
            dropout_mask,
        )
        # 4 tokens × 2 topk = 8 assignments
        assert flat_ids.numel() == T * topk
        assert start.numel() == n_local
        assert size.sum().item() == T * topk
        assert active.numel() == n_local  # both experts have ≥ 1 assignment
        assert (size > 0).all()

    def test_capacity_clipping(self):
        T, topk = 8, 4
        flat = torch.zeros(T, 4)
        # All 8 tokens route to expert 0 (topk=4, so 32 assignments, all to exp 0)
        indices = torch.zeros(T, topk, dtype=torch.long)
        weights = torch.full((T, topk), 0.25)
        n_local, experts_start, n_routed = 1, 0, 1
        # capacity = capacity_factor * expected = 1.0 * 32 = 32
        # → no clipping
        dropout_mask = torch.ones(n_local, dtype=torch.bool)
        _, _, _, size, _ = compute_routing_segments(
            T,
            topk,
            flat,
            indices,
            weights,
            n_local,
            experts_start,
            n_routed,
            1.0,
            dropout_mask,
        )
        assert size[0].item() == T * topk

        # capacity = 0.5 → 16 → clips to 16
        _, _, _, size_clipped, _ = compute_routing_segments(
            T,
            topk,
            flat,
            indices,
            weights,
            n_local,
            experts_start,
            n_routed,
            0.5,
            dropout_mask,
        )
        assert size_clipped[0].item() == 16


# ── DeepSeekMoE ────────────────────────────────────────────────────────────
class TestDeepSeekMoE:
    def _cfg(self) -> dict:
        return dict(
            dim=8,
            n_routed_experts=4,
            n_shared_experts=2,
            n_activated_experts=2,
            moe_inter_dim=16,
            expert_capacity_factor=1.5,
            expert_dropout_prob=0.0,
            moe_warmup_steps=0,
            n_expert_groups=2,
            n_limited_groups=1,
            group_topk=1,
            route_scale=1.0,
            bias_upper_threshold=0.1,
            bias_lower_threshold=0.1,
            moe_activation="swiglu",
        )

    def test_constructs_with_correct_local_expert_count(self):
        m = DeepSeekMoE(self._cfg(), world_size=2, rank=0)
        assert m.n_local_experts == 2  # 4 / 2
        assert m.experts_start == 0
        assert m.experts_end == 2

    def test_constructs_rank_1_shard(self):
        m = DeepSeekMoE(self._cfg(), world_size=2, rank=1)
        assert m.experts_start == 2
        assert m.experts_end == 4

    def test_world_size_must_divide_experts(self):
        cfg = self._cfg()
        cfg["n_routed_experts"] = 3
        with pytest.raises(ValueError, match="must be divisible"):
            DeepSeekMoE(cfg, world_size=2, rank=0)

    def test_load_balance_loss_returns_scalar(self):
        m = DeepSeekMoE(self._cfg(), world_size=1, rank=0)
        # Force a routing pass via a dummy forward
        x = torch.randn(4, 8)
        _ = m(x)  # populates _last_weights, _last_indices
        loss = m.get_load_balance_loss()
        assert loss.dim() == 0
        assert loss.item() >= 0.0
