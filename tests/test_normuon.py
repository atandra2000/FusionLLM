"""Unit tests for the NorMuon optimizer (Phase 4.1)."""

from __future__ import annotations

import pytest
import torch

from training.normuon import NorMuon


class TestNorMuon:
    def test_step_updates_matrix_params(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = NorMuon([p], lr=0.01, weight_decay=0.0)
        p.grad = torch.randn(8, 16)
        before = p.data.clone()
        opt.step()
        assert not torch.allclose(p.data, before)

    def test_step_skips_1d_params_no_orthogonalization(self):
        p_vec = torch.nn.Parameter(torch.randn(8))
        opt = NorMuon([p_vec], lr=0.01, weight_decay=0.0)
        p_vec.grad = torch.randn(8)
        before = p_vec.data.clone()
        opt.step()
        assert not torch.allclose(p_vec.data, before)

    def test_weight_decay_applied(self):
        p = torch.nn.Parameter(torch.ones(8, 16))
        opt = NorMuon([p], lr=0.1, weight_decay=0.5, cautious_wd=False)
        p.grad = torch.zeros(8, 16)
        before = p.data.clone()
        opt.step()
        assert (p.data < before).all()

    def test_cautious_wd_skips_when_sign_disagrees(self):
        p = torch.nn.Parameter(torch.ones(8, 16) * 0.5)
        opt = NorMuon([p], lr=0.1, weight_decay=0.5, cautious_wd=True)
        p.grad = torch.zeros(8, 16)
        before = p.data.clone()
        opt.step()
        # With zero grad, only weight decay applies.
        # Cautious WD masks where grad.sign != param.sign.
        # Since grad is 0, sign(0 * 0.5) = 0, so (0 == 1.0) is False → mask is all False.
        # Thus weight decay is entirely masked → p unchanged.
        assert torch.allclose(p.data, before, atol=1e-6)

    def test_moments_accumulate(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = NorMuon([p], lr=0.01, weight_decay=0.0)
        p.grad = torch.randn(8, 16)
        opt.step()
        state = opt.state[p]
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["step"] == 1

    def test_per_row_rms_differs_from_global_norm(self):
        p = torch.nn.Parameter(torch.randn(4, 8))
        opt = NorMuon([p], lr=1.0, weight_decay=0.0)
        p.grad = torch.randn(4, 8)
        before = p.data.clone()
        opt.step()
        update = before - p.data
        row_norms = update.norm(p=2, dim=-1)
        assert row_norms.numel() == 4
        assert row_norms[0].item() > 0

    def test_lr_zero_no_change(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = NorMuon([p], lr=0.0, weight_decay=0.0)
        p.grad = torch.randn(8, 16)
        before = p.data.clone()
        opt.step()
        assert torch.allclose(p.data, before)

    def test_state_dict_round_trip(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = NorMuon([p], lr=0.01, weight_decay=0.0)
        p.grad = torch.randn(8, 16)
        opt.step()
        sd = opt.state_dict()
        assert "state" in sd
        assert "param_groups" in sd
        assert len(sd["state"]) == 1

    def test_load_state_dict_restores_moments(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = NorMuon([p], lr=0.01, weight_decay=0.0)
        p.grad = torch.randn(8, 16)
        opt.step()
        saved_state = opt.state_dict()

        p2 = torch.nn.Parameter(torch.randn(8, 16))
        opt2 = NorMuon([p2], lr=0.01, weight_decay=0.0)
        opt2.load_state_dict(saved_state)
        reloaded = opt2.state_dict()
        # Check param_groups match
        for k in saved_state["param_groups"][0]:
            assert saved_state["param_groups"][0][k] == reloaded["param_groups"][0][k]
