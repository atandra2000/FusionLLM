"""Unit tests for `utils/logging.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `TrainerLogger` — constructor with backends disabled.
  * `log` — increments the EMA loss, prints to stdout.
  * `log_validation` — emits a `[VAL]` line.
  * `save_log` — appends JSONL to disk.
  * `log_summary` — no-op when backends are disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.logging import TrainerLogger


class TestTrainerLoggerNoBackends:
    def test_constructs_with_no_backends(self):
        lg = TrainerLogger(wandb_enabled=False)
        assert lg.wandb_enabled is False

    def test_log_emits_ema_and_prints(self, capsys):
        lg = TrainerLogger(
            log_interval=1,
            wandb_enabled=False,
        )
        lg.log(step=1, loss=2.5)
        out = capsys.readouterr().out
        assert "step=" in out
        assert "loss=" in out
        assert lg._ema_loss == 2.5

    def test_ema_is_smoothed(self):
        lg = TrainerLogger(
            log_interval=1,
            wandb_enabled=False,
        )
        lg.log(step=1, loss=2.0)
        lg.log(step=2, loss=4.0)
        # EMA should be between 2.0 and 4.0
        assert 2.0 < lg._ema_loss < 4.0

    def test_log_validation_prints(self, capsys):
        lg = TrainerLogger(wandb_enabled=False)
        lg.log_validation(step=10, val_loss=1.5, val_metrics={"acc": 0.42})
        out = capsys.readouterr().out
        assert "[VAL]" in out
        assert "val_loss=1.5" in out
        assert "acc=0.42" in out

    def test_save_log_writes_jsonl(self, tmp_path: Path):
        lg = TrainerLogger(wandb_enabled=False)
        p = tmp_path / "log.jsonl"
        lg.save_log(str(p), {"step": 1, "loss": 2.0})
        lg.save_log(str(p), {"step": 2, "loss": 1.9})
        lines = p.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[1])["loss"] == 1.9

    def test_log_summary_does_not_raise(self):
        lg = TrainerLogger(wandb_enabled=False)
        lg.log_summary({"final_loss": 1.0})  # no backends, should be a no-op

    def test_finish_does_not_raise(self):
        lg = TrainerLogger(wandb_enabled=False)
        lg.finish()
