"""Unit tests for `utils/checkpoint.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `CheckpointManager` — atomic save/load, async worker lifecycle.
  * `_atomic_save_safetensors` — round-trips a state dict.
  * `_checkpoint_complete` — True only when all required files exist.
  * `latest_step` / `list_checkpoints` — return the expected step.

DCP-backed resume and the FSDP2 save path land in Phase 3.2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from utils.checkpoint import CheckpointManager


def _toy_model() -> nn.Module:
    return nn.Linear(8, 16)


def _toy_optim(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


class TestAtomicSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        m = _toy_model()
        opt = _toy_optim(m)
        # Run a fake step so the optimizer state is non-empty
        x = torch.randn(2, 8)
        y = m(x).sum()
        y.backward()
        opt.step()

        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        ckpt.save(m, opt, step=100, val_loss=2.5, keep_last_n=None)

        m2 = _toy_model()
        opt2 = _toy_optim(m2)
        meta = ckpt.load(m2, step=100, device="cpu", optimizer=opt2)
        assert meta["step"] == 100
        assert "best_val_loss" in meta
        # Weights match
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_writes_meta_file(self, tmp_path: Path):
        m = _toy_model()
        opt = _toy_optim(m)
        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        ckpt.save(m, opt, step=10, val_loss=1.23, extra_meta={"foo": "bar"})
        meta_path = tmp_path / "meta_step_10.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["val_loss"] == 1.23
        assert meta.get("foo") == "bar"

    def test_checkpoint_complete(self, tmp_path: Path):
        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        # No checkpoint yet
        assert ckpt._checkpoint_complete(10) is False
        m = _toy_model()
        opt = _toy_optim(m)
        ckpt.save(m, opt, step=10, val_loss=None)
        assert ckpt._checkpoint_complete(10) is True

    def test_latest_step_returns_highest_complete(self, tmp_path: Path):
        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        m = _toy_model()
        opt = _toy_optim(m)
        for s in (10, 20, 30):
            ckpt.save(m, opt, step=s, val_loss=None)
        assert ckpt.latest_step() == 30
        assert ckpt.list_checkpoints() == [10, 20, 30]


class TestAsyncLifecycle:
    def test_async_worker_starts_and_stops(self, tmp_path: Path):
        ckpt = CheckpointManager(str(tmp_path), async_mode=True)
        assert ckpt._async_thread is not None
        assert ckpt._async_thread.is_alive()
        ckpt._stop_async_worker()
        assert ckpt._async_thread is None

    def test_save_async_runs_and_persists(self, tmp_path: Path):
        ckpt = CheckpointManager(str(tmp_path), async_mode=True)
        m = _toy_model()
        opt = _toy_optim(m)
        done = []

        def cb(err):
            done.append(err)

        ckpt.save_async(m, opt, step=42, callback=cb)
        # Give the worker a moment
        time.sleep(0.5)
        assert ckpt._checkpoint_complete(42)
        ckpt._stop_async_worker()

    def test_keep_last_n_prunes_old(self, tmp_path: Path):
        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        m = _toy_model()
        opt = _toy_optim(m)
        for s in (10, 20, 30, 40):
            ckpt.save(m, opt, step=s, val_loss=None)
        ckpt.keep_last_n(2)
        remaining = ckpt.list_checkpoints()
        assert remaining == [30, 40]


class TestBestValLoss:
    def test_best_val_loss_persists_across_init(self, tmp_path: Path):
        ck1 = CheckpointManager(str(tmp_path), async_mode=False)
        m = _toy_model()
        opt = _toy_optim(m)
        ck1.save(m, opt, step=10, val_loss=2.0)
        ck1.save(m, opt, step=20, val_loss=1.5)  # new best
        # Re-instantiate; the best val loss should be restored
        ck2 = CheckpointManager(str(tmp_path), async_mode=False)
        assert ck2.best_val_loss == 1.5
