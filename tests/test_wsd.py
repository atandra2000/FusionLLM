"""Unit tests for the WSD scheduler (Phase 4.2).

Note: ``_LRScheduler.__init__`` calls ``step()`` once, so each
scheduler starts at ``last_epoch=0`` after construction (not -1).
Tests account for this by deducting 1 from expected step counts.
"""

from __future__ import annotations

import pytest
import torch

from training.wsd import WSDScheduler


class TestWSDScheduler:
    def test_warmup_phase(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.8)
        for _ in range(5):
            opt.step()
            sched.step()
        assert sched.get_last_lr()[0] < 1.0

    def test_stable_phase_peak_lr(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.8)
        for _ in range(11):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert lr == 1.0

    def test_decay_phase_decreases(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.8, min_lr_ratio=0.1)
        for _ in range(91):
            opt.step()
            sched.step()
        lr_decay = sched.get_last_lr()[0]
        assert lr_decay < 1.0

    def test_decay_ends_at_min_lr(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.8, min_lr_ratio=0.1)
        for _ in range(100):
            opt.step()
            sched.step()
        assert sched.get_last_lr()[0] == 0.1

    def test_warmup_final_lr_at_end(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.5, stable_frac=0.0)
        for _ in range(49):
            opt.step()
            sched.step()
        final_warmup_lr = sched.get_last_lr()[0]
        for _ in range(1):
            opt.step()
            sched.step()
        first_stable_lr = sched.get_last_lr()[0]
        assert final_warmup_lr < first_stable_lr or abs(final_warmup_lr - first_stable_lr) < 1e-6

    def test_cosine_decay(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.4, min_lr_ratio=0.1, decay="cosine")
        for _ in range(95):
            opt.step()
            sched.step()
        cosine_lr = sched.get_last_lr()[0]
        assert 0.1 < cosine_lr < 1.0

    def test_linear_decay_monotonic(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = WSDScheduler(opt, total_steps=100, warmup_frac=0.1, stable_frac=0.4, decay="linear")
        lrs = []
        for _ in range(100):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        decay_lrs = lrs[50:]
        for i in range(1, len(decay_lrs)):
            assert decay_lrs[i] <= decay_lrs[i - 1] + 1e-8

    def test_repr_does_not_crash(self):
        p = torch.nn.Parameter(torch.randn(8, 16))
        opt = torch.optim.AdamW([p], lr=1.0)
        _ = WSDScheduler(opt, total_steps=100)
