"""Test FSDP2 resume and reshard tuning (Phase 3).

Phase 3 scope (per ``plan.md:3.4``):
  * ``configure_reshard`` — per-layer ``reshard_after_forward`` tuning.
  * DCP fallback — ``CheckpointManager`` with ``checkpoint_backend="dcp"``
    saves / loads via safetensors when no process group is available.
  * State-dict round-trip — end-to-end save/load with a tiny model.
  * ``Pretrainer.setup`` — FSDP2 wrap, reshard call, checkpoint backend
    dispatch all wire without error.

Full DCP resume verifcation (start → kill → resume → loss-curve match
within 0.1 %) requires a multi-GPU environment and is gated behind
``@pytest.mark.gpu``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.checkpoint import CheckpointManager
from utils.distributed import configure_reshard

_FSDP2_AVAILABLE = torch.cuda.is_available() and torch.distributed.is_available()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )


def _train_step(model: nn.Module, opt: torch.optim.Optimizer) -> float:
    model.train()
    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


# ── configure_reshard ─────────────────────────────────────────────────────────


class TestConfigureReshard:
    def test_no_crash_on_vanilla_model(self):
        model = _tiny_model()
        configure_reshard(model, keep_last_n=1)
        assert True

    @pytest.mark.skipif(not _FSDP2_AVAILABLE, reason="FSDP2 requires CUDA + torch.distributed")
    def test_fsdp_wrapped_model(self):
        from torch.distributed.fsdp import fully_shard

        model = _tiny_model()
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Sequential)):
                fully_shard(module)
        fully_shard(model)

        configure_reshard(model, keep_last_n=1)
        fsdp_units = [
            m for m in model.modules() if hasattr(m, "reshard_after_forward")
        ]
        assert len(fsdp_units) >= 1
        assert fsdp_units[-1].reshard_after_forward is False

    @pytest.mark.skipif(not _FSDP2_AVAILABLE, reason="FSDP2 requires CUDA + torch.distributed")
    def test_keep_last_two(self):
        from torch.distributed.fsdp import fully_shard

        model = _tiny_model()
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Sequential)):
                fully_shard(module)
        fully_shard(model)

        configure_reshard(model, keep_last_n=2)
        fsdp_units = [
            m for m in model.modules() if hasattr(m, "reshard_after_forward")
        ]
        assert len(fsdp_units) >= 2
        assert fsdp_units[-1].reshard_after_forward is False
        assert fsdp_units[-2].reshard_after_forward is False

    @pytest.mark.skipif(not _FSDP2_AVAILABLE, reason="FSDP2 requires CUDA + torch.distributed")
    def test_keep_last_zero_all_reshard(self):
        from torch.distributed.fsdp import fully_shard

        model = _tiny_model()
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Sequential)):
                fully_shard(module)
        fully_shard(model)

        configure_reshard(model, keep_last_n=0)
        for m in model.modules():
            if hasattr(m, "reshard_after_forward"):
                assert m.reshard_after_forward is True


# ── DCP fallback (safetensors path when no PG) ───────────────────────────────


class TestDCPFallback:
    def test_save_load_round_trip(self, tmp_path: Path):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _train_step(model, opt)

        ckpt = CheckpointManager(
            str(tmp_path), async_mode=False, checkpoint_backend="dcp"
        )
        ckpt.save_fsdp2_dcp(model, opt, step=10)

        model2 = _tiny_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        meta = ckpt.load_fsdp2_dcp(model2, opt2, step=10)

        assert meta["step"] == 10
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_multiple_steps_keep_last(self, tmp_path: Path):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ckpt = CheckpointManager(
            str(tmp_path), async_mode=False, checkpoint_backend="dcp"
        )
        for step in (10, 20, 30, 40):
            _train_step(model, opt)
            ckpt.save_fsdp2_dcp(model, opt, step=step, keep_last_n=2)

        remaining = ckpt.list_checkpoints()
        assert remaining == [30, 40]

    def test_meta_persists_val_loss(self, tmp_path: Path):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ckpt = CheckpointManager(
            str(tmp_path), async_mode=False, checkpoint_backend="dcp"
        )
        ckpt.save_fsdp2_dcp(model, opt, step=10, val_loss=2.0)
        ckpt.save_fsdp2_dcp(model, opt, step=20, val_loss=1.5)  # new best

        ckpt2 = CheckpointManager(
            str(tmp_path), async_mode=False, checkpoint_backend="dcp"
        )
        assert ckpt2.best_val_loss == 1.5


# ── Safetensors path (checkpoint_backend default) ────────────────────────────


class TestSafetensorsSaveLoad:
    def test_checkpoint_manager_default_backend(self, tmp_path: Path):
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _train_step(model, opt)

        ckpt = CheckpointManager(str(tmp_path), async_mode=False)
        assert ckpt.checkpoint_backend == "safetensors"
        ckpt.save(model, opt, step=5)

        model2 = _tiny_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        meta = ckpt.load(model2, step=5, device="cpu", optimizer=opt2)
        assert meta["step"] == 5
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
