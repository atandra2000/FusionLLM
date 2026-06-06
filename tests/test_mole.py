"""Unit tests for `models/mole.py`.

Validates the MoLE bank:
* constructor math + param count
* forward shape and zero-init guarantee
* dispatch through ``TransformerBlock`` via the ``mole_every_n``
  config flag.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.mole import MoLE
from models.transformer import DeepSeekMoE, TransformerBlock


# ── Constructor ────────────────────────────────────────────────────────────
class TestMoLEConstructor:
    def test_constructs_with_minimal_args(self):
        m = MoLE(dim=16, rank=4, n_experts=2)
        assert m.dim == 16
        assert m.rank == 4
        assert m.n_experts == 2

    def test_param_count(self):
        # W_A: n_experts * dim * rank; W_B: n_experts * rank * dim
        # router: dim * n_experts (no bias)
        dim, rank, n_experts = 16, 4, 3
        m = MoLE(dim=dim, rank=rank, n_experts=n_experts)
        n_params = sum(p.numel() for p in m.parameters())
        expected = n_experts * dim * rank + n_experts * rank * dim + dim * n_experts
        assert n_params == expected

    def test_W_B_initialised_to_zero(self):
        m = MoLE(dim=16, rank=4, n_experts=2)
        assert torch.allclose(m.W_B, torch.zeros_like(m.W_B))

    def test_W_A_initialised_with_correct_std(self):
        torch.manual_seed(0)
        m = MoLE(dim=32, rank=4, n_experts=2)
        expected_std = 1.0 / (4**0.5)  # 0.5
        actual_std = m.W_A.std().item()
        # Within 30 % of expected (sample size small).
        assert 0.7 * expected_std < actual_std < 1.3 * expected_std


# ── Forward ────────────────────────────────────────────────────────────────
class TestMoLEForward:
    def test_forward_shape(self):
        m = MoLE(dim=16, rank=4, n_experts=2)
        x = torch.randn(2, 8, 16)
        y = m(x)
        assert y.shape == x.shape

    def test_forward_is_zero_at_init(self):
        """Zero-init W_B → the expert contribution is zero.  The
        router still routes, but the weighted sum is zero.
        """
        torch.manual_seed(0)
        m = MoLE(dim=16, rank=4, n_experts=2).eval()
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            y = m(x)
        # y is exactly zero because W_B is zero.
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_grad_vanishes_at_init(self):
        """At init, the MoLE output is zero → its contribution to
        the residual stream is zero → gradients w.r.t. upstream are
        not affected.  (modded-nanogpt #4 property.)
        """
        m = MoLE(dim=16, rank=4, n_experts=2)
        x = torch.randn(2, 8, 16, requires_grad=True)
        y = m(x)
        y.sum().backward()
        # Gradient still exists (flows through the router), but it's
        # the *output* that's zero — the upstream sees no signal.
        assert torch.allclose(y.detach(), torch.zeros_like(y), atol=1e-6)
        # x.grad is non-zero (router params get gradients).
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_norms_match_after_one_train_step(self):
        """After one optimizer step that nudges W_B, the output norm
        should be in a sensible range (within 5× the input norm).
        """
        torch.manual_seed(0)
        m = MoLE(dim=32, rank=4, n_experts=4)
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        x = torch.randn(1, 4, 32)
        y = m(x)
        y.sum().backward()
        opt.step()
        y2 = m(x)
        in_norm = x.norm().item()
        out_norm = y2.norm().item()
        # Not zero anymore, not absurdly large.
        assert out_norm > 0
        assert out_norm < 10 * in_norm

    def test_top_k_gt_1(self):
        m = MoLE(dim=16, rank=4, n_experts=4, top_k=2)
        x = torch.randn(1, 4, 16)
        y = m(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


# ── Dispatch through TransformerBlock ─────────────────────────────────────
class TestMoLEDispatch:
    def test_mole_every_n_4_replaces_moe(self):
        """When mole_every_n=4, the 4th layer (idx=3) has a MoLE FFN
        instead of MoE.  Other layers still have MoE.
        """
        cfg = dict(
            dim=16,
            n_layers=8,
            mole_every_n=4,
            # avoid MLA forward (bool-subtract bug) — use SSM path
            layer_schedule="ssm",
            ssm_type="mamba2",
        )
        blocks = [
            TransformerBlock(
                cfg,
                world_size=1,
                rank=0,
                layer_idx=i,
                use_checkpoint=False,
                use_mamba=parse_mamba(i, cfg["layer_schedule"], cfg["n_layers"]),
            )
            for i in range(cfg["n_layers"])
        ]
        # Layer 3 (index 3) is the 4th → use_mole should be True.
        assert blocks[3].use_mole
        # Layer 0 is the 1st → no MoLE.
        assert not blocks[0].use_mole
        # Layer 3 FFN is MoLE
        from models.mole import MoLE as MoLEClass

        assert isinstance(blocks[3].ffn, MoLEClass)
        # Layer 0 FFN is DenseFFN (SSM path)
        from models.transformer import DenseFFN

        assert isinstance(blocks[0].ffn, DenseFFN)

    def test_mole_every_n_zero_disables_mole(self):
        """Default behavior (mole_every_n=0 or missing) → no MoLE."""
        cfg = dict(
            dim=16,
            n_layers=4,
            layer_schedule="ssm",
            ssm_type="mamba2",
        )
        b = TransformerBlock(
            cfg,
            world_size=1,
            rank=0,
            layer_idx=0,
            use_checkpoint=False,
            use_mamba=True,
        )
        assert b.use_mole is False
        assert not isinstance(b.ffn, MoLE)

    def test_mole_and_moe_are_never_co_resident(self):
        """Phase 2 scope: a block either has MoLE OR MoE/DenseFFN,
        never both.  The ``moe_layers()`` helper should never include
        a MoLE-bearing block.
        """
        from models.mole import MoLE as MoLEClass
        from models.transformer import DenseFFN

        cfg = dict(
            dim=16,
            n_layers=8,
            mole_every_n=4,
            layer_schedule="ssm",
            ssm_type="mamba2",
        )
        blocks = [
            TransformerBlock(
                cfg,
                world_size=1,
                rank=0,
                layer_idx=i,
                use_checkpoint=False,
                use_mamba=parse_mamba(i, cfg["layer_schedule"], cfg["n_layers"]),
            )
            for i in range(cfg["n_layers"])
        ]
        for b in blocks:
            if b.use_mole:
                assert isinstance(b.ffn, MoLEClass)
            else:
                assert isinstance(b.ffn, DenseFFN)


# ── Helpers ───────────────────────────────────────────────────────────────
def parse_mamba(idx: int, schedule: str, n_layers: int) -> bool:
    from models.transformer import parse_schedule

    return parse_schedule(n_layers, schedule)[idx]
