"""Tests for Phase 6.2 — RunsCsvLogger and run_lm_eval wrapper."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.logging import RunsCsvLogger


class TestRunsCsvLogger:
    def test_creates_file_on_first_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            logger = RunsCsvLogger(str(path))
            logger.log(step=0, loss=2.5, ppl=12.18)
            assert path.exists()
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["step"] == "0"
            assert rows[0]["loss"] == "2.5"
            assert rows[0]["ppl"] == "12.18"

    def test_appends_multiple_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            logger = RunsCsvLogger(str(path))
            logger.log(step=0, loss=2.5, ppl=12.18)
            logger.log(step=1000, loss=1.5, ppl=4.48)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[1]["step"] == "1000"
            assert rows[1]["loss"] == "1.5"

    def test_extra_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            logger = RunsCsvLogger(str(path))
            logger.log(step=0, loss=2.5, ppl=12.18, hellaswag=0.42, arc_c=0.35)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["hellaswag"] == "0.42"
            assert rows[0]["arc_c"] == "0.35"

    def test_header_only_written_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            logger = RunsCsvLogger(str(path))
            logger.log(step=0, loss=2.5, ppl=12.18)
            logger.log(step=1000, loss=1.5, ppl=4.48)
            with open(path) as f:
                content = f.read()
            assert content.count("step,loss,ppl") == 1

    def test_reuses_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            path.write_text("step,loss,ppl\n0,2.5,12.18\n")
            logger = RunsCsvLogger(str(path))
            assert logger._header_written
            logger.log(step=1000, loss=1.5, ppl=4.48)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2


class TestRunLmEvalWrapper:
    """run_lm_eval should gracefully return None when lm_eval is not installed."""

    def test_returns_none_on_cpu(self):
        from eval.run_lm_eval import run_lm_eval

        model = nn.Linear(10, 10)
        result = run_lm_eval(model, device="cpu")
        assert result is None

    def test_returns_none_without_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("requires CPU-only environment")
        from eval.run_lm_eval import run_lm_eval

        model = nn.Linear(10, 10)
        result = run_lm_eval(model, device="cuda")
        assert result is None

    def test_import_does_not_crash(self):
        from eval.run_lm_eval import run_lm_eval

        assert callable(run_lm_eval)
