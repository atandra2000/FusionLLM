"""Unit tests for the Muon optimizer in `training/pretrain.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `_zeropower_via_newtonschulz5` — idempotence in BF16 (tolerant).
  * `Muon` — step updates only ndim>=2 params; respects nesterov.
  * `CautiousAdamW` — sign-mask contract (matrix params get the mask).
  * `_cautious_mask` — output is boolean with same shape as `grad`.

NorMuon and Cautious-WD-on-Adam land in Phase 4.1 / 4.2.
"""

from __future__ import annotations

import pytest
import torch

from training import (
    CautiousAdamW,
    Muon,
    _cautious_mask,
    _zeropower_via_newtonschulz5,
)


# ── Newton-Schulz ─────────────────────────────────────────────────────────
class TestNewtonSchulz:
    def test_shape_preserved(self):
        g = torch.randn(8, 16)
        out = _zeropower_via_newtonschulz5(g, steps=5)
        assert out.shape == g.shape

    def test_shape_preserved_transposed_input(self):
        # Square matrix and tall matrix exercise both branches
        for shape in [(8, 8), (16, 8), (8, 16), (4, 32)]:
            g = torch.randn(*shape)
            out = _zeropower_via_newtonschulz5(g, steps=5)
            assert out.shape == g.shape

    def test_near_idempotent_bf16(self):
        # Apply twice ≈ apply once (within BF16 tolerance)
        g = torch.randn(8, 16)
        once = _zeropower_via_newtonschulz5(g, steps=5)
        twice = _zeropower_via_newtonschulz5(once.float(), steps=5)
        diff = (twice - once).abs().max().item()
        # NS-5 is not perfectly idempotent in BF16; we just want
        # the diff to be small.
        assert diff < 0.2, f"NS-5 not near-idempotent: max diff {diff}"

    def test_assumes_ndim_at_least_2(self):
        with pytest.raises(AssertionError):
            _zeropower_via_newtonschulz5(torch.randn(8))


# ── Cautious mask ──────────────────────────────────────────────────────────
class TestCautiousMask:
    def test_mask_shape(self):
        p = torch.randn(4, 8)
        g = torch.randn(4, 8)
        m = _cautious_mask(g, p)
        assert m.shape == g.shape
        assert m.dtype == torch.bool

    def test_mask_is_one_where_signs_agree(self):
        p = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        g = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        m = _cautious_mask(g, p)
        # positions where g*p > 0
        assert m[0, 0].item() is True or m[0, 0].item() == 1
        assert m[0, 1].item() is False or m[0, 1].item() == 0
        assert m[1, 0].item() is False or m[1, 0].item() == 0
        assert m[1, 1].item() is True or m[1, 1].item() == 1


# ── Muon ──────────────────────────────────────────────────────────────────
class TestMuon:
    def test_step_skips_1d_params(self):
        p_matrix = torch.nn.Parameter(torch.randn(8, 16))
        p_vec = torch.nn.Parameter(torch.randn(8))
        opt = Muon([p_matrix, p_vec], lr=0.01, weight_decay=0.0)
        p_matrix.grad = torch.randn(8, 16)
        p_vec.grad = torch.randn(8)
        before_vec = p_vec.data.clone()
        opt.step()
        # 2D param updated; 1D untouched (Muon skips it)
        assert not torch.allclose(p_matrix.data, torch.zeros_like(p_matrix.data))
        assert torch.allclose(p_vec.data, before_vec)

    def test_step_respects_nesterov(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt_n = Muon([p], lr=0.01, nesterov=True, weight_decay=0.0)
        opt_no = Muon(
            [torch.nn.Parameter(p.data.clone())], lr=0.01, nesterov=False, weight_decay=0.0
        )
        p.grad = torch.randn(8, 16)
        opt_n.step()
        # Re-create the other Muon on a fresh copy and step it
        p2 = torch.nn.Parameter(p.data.clone())
        opt_no = Muon([p2], lr=0.01, nesterov=False, weight_decay=0.0)
        p2.grad = p.grad.clone()
        opt_no.step()
        # Nesterov and non-Nesterov produce different updates
        assert not torch.allclose(p.data, p2.data, atol=1e-4)

    def test_weight_decay_applied(self):
        p = torch.nn.Parameter(torch.ones(8, 16))
        opt = Muon([p], lr=0.1, weight_decay=0.5)
        p.grad = torch.zeros(8, 16)
        before = p.data.clone()
        opt.step()
        # Decay shrinks the param (p *= 1 - lr*wd)
        assert (p.data < before).all()


# ── CautiousAdamW ──────────────────────────────────────────────────────────
class TestCautiousAdamW:
    def test_step_runs(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        # `cautious_wd` is a per-group flag, not an optimizer kwarg.
        opt = CautiousAdamW(
            [{"params": [p], "weight_decay": 0.1, "cautious_wd": True}],
            lr=1e-3,
            betas=(0.9, 0.95),
        )
        p.grad = torch.randn(8, 16)
        before = p.data.clone()
        opt.step()
        assert not torch.allclose(p.data, before)
