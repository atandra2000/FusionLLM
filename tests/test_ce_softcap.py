"""Unit tests for the fused CE + softcap kernel (Phase 4.4)."""

from __future__ import annotations

import pytest
import torch

from kernels.ce_softcap import ce_softcap, fused_ce_softcap, softcap


class TestSoftcap:
    def test_clamps_logits(self):
        logits = torch.tensor([[-100.0, 0.0, 100.0]])
        capped = softcap(logits, softcap_value=15.0)
        assert capped[0, 0].item() >= -15.0
        assert capped[0, 2].item() <= 15.0

    def test_zero_center_preserved(self):
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        capped = softcap(logits, softcap_value=15.0)
        assert torch.allclose(capped, torch.zeros_like(capped), atol=1e-6)


class TestCeSoftcap:
    def test_loss_shape(self):
        logits = torch.randn(2, 4, 8)
        targets = torch.randint(0, 8, (2, 4))
        loss = ce_softcap(logits, targets, softcap_value=15.0)
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        logits = torch.randn(2, 4, 16)
        targets = torch.randint(0, 16, (2, 4))
        loss = ce_softcap(logits, targets, softcap_value=15.0)
        assert torch.isfinite(loss).item()

    def test_ignore_index_works(self):
        logits = torch.randn(2, 4, 8)
        targets = torch.tensor([[0, -100, 1, -100], [2, 3, -100, 4]], dtype=torch.long)
        loss = ce_softcap(logits, targets, softcap_value=15.0, ignore_index=-100)
        assert torch.isfinite(loss).item()

    def test_higher_logits_for_correct_class_give_lower_loss(self):
        B, T, V = 2, 4, 16
        targets = torch.randint(0, V, (B, T))
        logits_bad = torch.randn(B, T, V) * 10
        logits_good = logits_bad.clone()
        for b in range(B):
            for t in range(T):
                logits_good[b, t, targets[b, t]] = 50.0
        loss_bad = ce_softcap(logits_bad, targets, softcap_value=50.0)
        loss_good = ce_softcap(logits_good, targets, softcap_value=50.0)
        assert loss_good.item() <= loss_bad.item()


class TestFusedCeSoftcap:
    def test_fallback_matches_cpu(self):
        logits = torch.randn(2, 4, 16)
        targets = torch.randint(0, 16, (2, 4))
        ref = ce_softcap(logits, targets, softcap_value=15.0)
        fused = fused_ce_softcap(logits, targets, softcap_value=15.0)
        assert torch.allclose(ref, fused, atol=1e-5)

    def test_ignore_index_matches(self):
        logits = torch.randn(2, 4, 16)
        targets = torch.tensor([[0, -100, 1, -100], [2, 3, -100, 4]], dtype=torch.long)
        ref = ce_softcap(logits, targets, softcap_value=15.0, ignore_index=-100)
        fused = fused_ce_softcap(logits, targets, softcap_value=15.0, ignore_index=-100)
        assert torch.allclose(ref, fused, atol=1e-5)
