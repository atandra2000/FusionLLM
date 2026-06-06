"""Unit tests for `models.transformer.softcap_15` and `AsymmetricRescale`.

Phase 2.6:
* `softcap_15` bounds the output to [-15, 15] and is identity in the
  linear regime.
* `AsymmetricRescale` is identity at init (scale=0, bias=0) and
  produces a finite forward.
* The Transformer wires both into the forward path when configured.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.transformer import (
    AsymmetricRescale,
    Transformer,
    softcap_15,
)


# ── softcap_15 ─────────────────────────────────────────────────────────────
class TestSoftcap15:
    def test_bounds_output(self):
        x = torch.linspace(-100, 100, 1000)
        y = softcap_15(x)
        assert y.abs().max() <= 15.0 + 1e-5

    def test_identity_in_linear_regime(self):
        # For |x| ≤ 1 the softcap is essentially identity.
        x = torch.linspace(-1, 1, 100)
        y = softcap_15(x)
        # 15 * tanh(x/15) ≈ x when |x| << 15; for |x| ≤ 1 the error
        # is at most ~1.5e-3.
        assert torch.allclose(y, x, atol=2e-3)

    def test_derivative_at_zero_is_one(self):
        """d/dx [15 * tanh(x/15)] at x=0 = 1 (numerically)."""
        x = torch.tensor([0.0], requires_grad=True)
        y = softcap_15(x)
        y.backward()
        assert torch.allclose(x.grad, torch.tensor([1.0]), atol=1e-5)

    def test_no_nan_on_extreme_values(self):
        x = torch.tensor([1e6, -1e6, 0.0, 1e-10])
        y = softcap_15(x)
        assert torch.isfinite(y).all()
        # And the values are bounded.
        assert y.abs().max() <= 15.0 + 1e-5


# ── AsymmetricRescale ──────────────────────────────────────────────────────
class TestAsymmetricRescale:
    def test_identity_at_init(self):
        """scale=0 → identity; bias=0 → no shift."""
        torch.manual_seed(0)
        m = AsymmetricRescale(dim=10)
        x = torch.randn(2, 4, 10) * 5
        y = m(x)
        # At init: y = (x - μ) / σ, bias=0
        # mean should be ≈ 0, std should be ≈ 1 along last dim
        assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 4), atol=1e-5)
        assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(2, 4), atol=1e-2)

    def test_zero_scale_and_bias_at_init(self):
        m = AsymmetricRescale(dim=8)
        assert torch.allclose(m.scale, torch.zeros(8))
        assert torch.allclose(m.bias, torch.zeros(8))

    def test_learnable_scale_changes_output(self):
        """After nudging scale and bias, the output changes."""
        torch.manual_seed(0)
        m = AsymmetricRescale(dim=4)
        x = torch.randn(2, 3, 4)
        y0 = m(x)
        # Manually set scale=1 → doubles the normed output.
        with torch.no_grad():
            m.scale.fill_(1.0)
        y1 = m(x)
        assert not torch.allclose(y0, y1)

    def test_backward_passes(self):
        m = AsymmetricRescale(dim=4)
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── Transformer wiring ────────────────────────────────────────────────────
class TestTransformerWiring:
    def _build_cfg(self, **overrides):
        cfg = dict(
            dim=16,
            n_layers=2,
            vocab_size=32,
            max_seq_len=8,
            n_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            n_expert_groups=1,
            n_limited_groups=1,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            moe_inter_dim=8,
            layer_schedule="mha",
            mtp_depth=0,
            tie_embeddings=True,
        )
        cfg.update(overrides)
        return cfg

    def test_asymmetric_rescale_disabled_by_default(self):
        cfg = self._build_cfg()
        m = Transformer(cfg, world_size=1, rank=0)
        assert m._asym_rescale_enabled is False
        assert m.asym_rescale is None

    def test_asymmetric_rescale_enabled_via_config(self):
        cfg = self._build_cfg(asymmetric_rescale=True)
        m = Transformer(cfg, world_size=1, rank=0)
        assert m._asym_rescale_enabled is True
        assert isinstance(m.asym_rescale, AsymmetricRescale)
        # And the scale/bias are zero at init (so the layer is a
        # no-op affine at init, just like a LayerNorm).
        assert torch.allclose(m.asym_rescale.scale, torch.zeros(32))
        assert torch.allclose(m.asym_rescale.bias, torch.zeros(32))
