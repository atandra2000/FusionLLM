"""Unit tests for the FSDP2 helpers in `utils/distributed.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `setup_distributed` / `cleanup_distributed` — single-process no-op.
  * `is_main_process` — True on rank 0, False otherwise.
  * `all_reduce_mean` — mean of values across ranks.
  * `barrier` — no-op when distributed is not initialised.

The full FSDP2 wrap test (`wrap_fsdp2`) requires a multi-GPU
environment and is marked `@pytest.mark.distributed`.
"""

from __future__ import annotations

import torch

from utils.distributed import (
    all_reduce_mean,
    barrier,
    is_main_process,
    setup_distributed,
)


class TestSingleProcess:
    def test_setup_distributed_returns_singletons(self):
        ws, rk, lr = setup_distributed()
        assert ws == 1
        assert rk == 0
        assert lr == 0

    def test_is_main_process_true_at_rank_0(self):
        assert is_main_process() is True

    def test_all_reduce_mean_no_op(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = all_reduce_mean(t)
        assert torch.allclose(out, t)

    def test_barrier_no_op(self):
        # Should not raise
        barrier()

    def test_setup_distributed_uses_local_rank_env(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        ws, rk, lr = setup_distributed()
        assert (ws, rk, lr) == (1, 0, 0)
