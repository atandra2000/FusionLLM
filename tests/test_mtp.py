"""Unit tests for `models/mtp.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `MTPBlock` — constructor shape, forward contract.
  * `MultiTokenPrediction` — depth=0 returns empty mtp_modules.
  * Shared-head injection (set_output_head).

Phase 2.4 additions:
  * depth=3 returns 3 (logits, target) pairs; the 3rd is shifted
    by 4 tokens.
  * softcap_ce bounds the loss.
  * mtp_loss_weight_schedule returns the right per-depth weights.
  * compute_mtp_loss is weighted correctly and supports softcap.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.mtp import (
    MTPBlock,
    MTPModule,
    MultiTokenPrediction,
    mtp_loss_weight_schedule,
    softcap_ce,
)


# ── MTPBlock ───────────────────────────────────────────────────────────────
class TestMTPBlock:
    def test_constructor_shapes(self):
        b = MTPBlock(dim=8, n_heads=2, inter_dim=16)
        assert b.proj.in_features == 16 and b.proj.out_features == 8
        assert b.w1.in_features == 8 and b.w1.out_features == 16
        assert b.w2.in_features == 16 and b.w2.out_features == 8

    def test_forward_shape(self):
        b = MTPBlock(dim=8, n_heads=2, inter_dim=16)
        prev = torch.randn(2, 4, 8)
        emb = torch.randn(2, 4, 8)
        y = b(prev, emb)
        assert y.shape == prev.shape


# ── MTPModule ──────────────────────────────────────────────────────────────
class TestMTPModule:
    def test_output_head_required(self):
        m = MTPModule(dim=8, n_heads=2, inter_dim=16, depth=1)
        prev = torch.randn(1, 4, 8)
        with pytest.raises(RuntimeError, match="output_head not set"):
            m(prev, prev)

    def test_shape_mismatch_raises(self):
        m = MTPModule(dim=8, n_heads=2, inter_dim=16, depth=1)
        m.set_output_head(torch.nn.Linear(8, 32, bias=False))
        prev = torch.randn(1, 4, 8)
        emb = torch.randn(1, 5, 8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            m(prev, emb)


# ── MultiTokenPrediction ───────────────────────────────────────────────────
class TestMultiTokenPrediction:
    def test_depth_zero_creates_no_modules(self):
        cfg = dict(dim=8, n_heads=2, inter_dim=16, mtp_depth=0, mtp_loss_weight=0.0)
        # main_model is a stand-in — MultiTokenPrediction only reads
        # `.embed` and `.head`; mock them
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(32, 8)
        main.head = torch.nn.Linear(8, 32, bias=False)
        mtp = MultiTokenPrediction(cfg, main_model=main)
        assert len(mtp.mtp_modules) == 0

    def test_depth_creates_one_module_per_depth(self):
        cfg = dict(dim=8, n_heads=2, inter_dim=16, mtp_depth=2, mtp_loss_weight=0.1)
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(32, 8)
        main.head = torch.nn.Linear(8, 32, bias=False)
        mtp = MultiTokenPrediction(cfg, main_model=main)
        assert len(mtp.mtp_modules) == 2
        # Output head is the main head (parameter reference, not copy)
        for m in mtp.mtp_modules:
            assert m.output_head is main.head

    def test_reinject_heads_after_load(self):
        cfg = dict(dim=8, n_heads=2, inter_dim=16, mtp_depth=1, mtp_loss_weight=0.1)
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(32, 8)
        main.head = torch.nn.Linear(8, 32, bias=False)
        mtp = MultiTokenPrediction(cfg, main_model=main)
        # Simulate loading a state dict that doesn't touch the head ref
        mtp.load_state_dict(mtp.state_dict())
        for m in mtp.mtp_modules:
            assert m.output_head is main.head


# ── Phase 2.4: softcap_ce ──────────────────────────────────────────────────
class TestSoftcapCE:
    def test_loss_is_capped(self):
        """The loss is bounded above by ``cap``."""
        # Random logits → cross-entropy can be large if logits are
        # miscalibrated.  softcap_ce should never exceed cap.
        torch.manual_seed(0)
        logits = torch.randn(8, 100) * 100  # very miscalibrated
        target = torch.randint(0, 100, (8,))
        cap = 5.0
        loss = softcap_ce(logits, target, cap=cap)
        assert loss.item() <= cap + 1e-5

    def test_loss_is_uncapped_when_below_cap(self):
        """When the raw CE is well below cap, softcap_ce ≈ raw CE."""
        torch.manual_seed(0)
        logits = torch.randn(8, 100) * 0.1  # well-calibrated
        target = torch.randint(0, 100, (8,))
        cap = 100.0  # huge cap
        loss = softcap_ce(logits, target, cap=cap)
        # Should be close to the raw CE (tanh(x/cap) ≈ x/cap when
        # raw << cap, so softcap_ce ≈ raw here).
        raw = nn.functional.cross_entropy(logits, target)
        assert torch.allclose(loss, raw, atol=1e-2)

    def test_ignore_index_excluded(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 10)
        target = torch.tensor([3, -100])
        loss = softcap_ce(logits, target, cap=100.0)
        # The loss should be well-defined (not NaN) and finite.
        assert torch.isfinite(loss)


# ── Phase 2.4: mtp_loss_weight_schedule ──────────────────────────────────
class TestMTPWeightSchedule:
    def test_default_schedule_is_linear_decreasing(self):
        ws = mtp_loss_weight_schedule(3)
        # Default: [0.3, 0.2, 0.1]
        assert ws[0] == pytest.approx(0.3)
        assert ws[-1] == pytest.approx(0.1)
        assert ws[1] == pytest.approx(0.2)

    def test_depth_zero_returns_empty(self):
        assert mtp_loss_weight_schedule(0) == []

    def test_depth_one_returns_single(self):
        ws = mtp_loss_weight_schedule(1)
        assert ws == [0.3]

    def test_custom_schedule(self):
        ws = mtp_loss_weight_schedule(2, schedule=[0.5, 0.5])
        assert ws == [0.5, 0.5]

    def test_custom_schedule_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            mtp_loss_weight_schedule(3, schedule=[0.1, 0.1])


# ── Phase 2.4: depth=3 alignment ─────────────────────────────────────────
class TestDepth3Alignment:
    def test_depth_3_returns_3_pairs(self):
        """With mtp_depth=3, the forward pass returns 3 (logits, target) pairs.
        The 3rd pair's target is shifted by 4 tokens (depth+1).
        """
        cfg = dict(
            dim=8,
            n_heads=2,
            inter_dim=16,
            mtp_depth=3,
            mtp_loss_weight=0.3,
        )
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(16, 8)
        main.head = torch.nn.Linear(8, 16, bias=False)

        # Build a minimal stand-in for main_model.forward_with_hidden
        class FakeMain(nn.Module):
            def __init__(self, embed, head):
                super().__init__()
                self.embed = embed
                self.head = head

            def forward_with_hidden(self, tokens, start_pos, use_cache):
                x = self.embed(tokens)
                # Pad hidden to (b, t, dim) — fake.
                return self.head(x), x

        fm = FakeMain(main.embed, main.head)
        mtp = MultiTokenPrediction(cfg, main_model=fm)
        assert len(mtp.mtp_modules) == 3

        tokens = torch.randint(0, 16, (2, 12))
        main_logits, mtp_pairs, prev_h = mtp(tokens)
        # 3 pairs as long as seq_len - depth - 1 > 0 for each depth.
        # depth=1: usable = 12-1-1 = 10
        # depth=2: usable = 12-2-1 = 9
        # depth=3: usable = 12-3-1 = 8
        assert len(mtp_pairs) == 3
        logits, tgt = mtp_pairs[0]
        assert logits.shape[1] == 10
        assert tgt.shape[1] == 10
        logits, tgt = mtp_pairs[1]
        assert logits.shape[1] == 9
        assert tgt.shape[1] == 9
        logits, tgt = mtp_pairs[2]
        assert logits.shape[1] == 8
        assert tgt.shape[1] == 8

    def test_default_mtp_depth_is_3(self):
        cfg = dict(dim=8, n_heads=2, inter_dim=16, mtp_loss_weight=0.3)
        # No mtp_depth in cfg → default 3.
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(16, 8)
        main.head = torch.nn.Linear(8, 16, bias=False)
        mtp = MultiTokenPrediction(cfg, main_model=main)
        assert mtp.depth == 3
        assert len(mtp.mtp_modules) == 3


# ── Phase 2.4: compute_mtp_loss ──────────────────────────────────────────
class TestComputeMTPLoss:
    def _build_mtp(self, softcap: bool = True):
        cfg = dict(
            dim=8,
            n_heads=2,
            inter_dim=16,
            mtp_depth=2,
            mtp_loss_weight=0.3,
            mtp_softcap=softcap,
            mtp_softcap_value=10.0,
        )
        main = torch.nn.Module()
        main.embed = torch.nn.Embedding(16, 8)
        main.head = torch.nn.Linear(8, 16, bias=False)

        class FakeMain(nn.Module):
            def __init__(self, embed, head):
                super().__init__()
                self.embed = embed
                self.head = head

            def forward_with_hidden(self, tokens, start_pos, use_cache):
                x = self.embed(tokens)
                return self.head(x), x

        return MultiTokenPrediction(cfg, main_model=FakeMain(main.embed, main.head))

    def test_empty_pairs_returns_zero(self):
        mtp = self._build_mtp()
        loss = mtp.compute_mtp_loss([])
        assert loss.item() == 0.0

    def test_softcap_loss_is_capped(self):
        mtp = self._build_mtp(softcap=True)
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 16) * 100
        target = torch.randint(0, 16, (1, 4))
        # Weight is 0.3 for depth 0 and 0.2 for depth 1 → sum 0.5; cap=10
        loss = mtp.compute_mtp_loss([(logits, target), (logits, target)])
        # Each depth's CE is capped at 10; weighted sum 0.5 * 10 = 5
        assert loss.item() <= 5.0 + 1e-3

    def test_raw_loss_when_softcap_off(self):
        mtp = self._build_mtp(softcap=False)
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 16) * 0.1  # well-calibrated
        target = torch.randint(0, 16, (1, 4))
        loss = mtp.compute_mtp_loss([(logits, target)])
        raw = nn.functional.cross_entropy(logits.view(-1, 16), target.view(-1))
        # weight is 0.3 for depth 0
        assert torch.allclose(loss, 0.3 * raw, atol=1e-4)
