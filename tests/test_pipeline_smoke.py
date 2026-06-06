"""End-to-end pipeline smoke test (Phase 6.4).

Runs ``scripts/run_smoke_7b.sh`` as a subprocess with a tiny model
config (``configs/smoke_pretrain.yaml``, 10 steps, 1 GPU) and asserts
that the process exits 0 and a checkpoint file is written.

This test requires a CUDA-capable GPU and is gated behind
``@pytest.mark.gpu``.  It is **not** the benchmark — it only verifies
that the full Pretrainer pipeline (model init, FSDP2 wrap, data loading,
optimization, eval, checkpointing) completes without error.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu


SMOKE_CONFIG = "configs/smoke_pretrain.yaml"
CHECKPOINT_DIR = "checkpoints/smoke"
SCRIPT = "scripts/run_smoke_7b.sh"


def _find_checkpoint(dir_path: Path) -> bool:
    """Return True if at least one checkpoint file exists under *dir_path*."""
    if not dir_path.is_dir():
        return False
    safetensors = list(dir_path.rglob("*.safetensors"))
    pt = list(dir_path.rglob("*.pt"))
    return len(safetensors) > 0 or len(pt) > 0


@pytest.fixture(scope="module", autouse=True)
def _clean_checkpoints():
    """Remove any leftover checkpoints before the test."""
    ckpt_dir = Path(CHECKPOINT_DIR)
    if ckpt_dir.is_dir():
        import shutil

        shutil.rmtree(ckpt_dir)
    yield


def test_pipeline_smoke():
    assert torch.cuda.is_available(), "GPU required for pipeline smoke test"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(
        [sys.executable, "training/pretrain.py", "--config", SMOKE_CONFIG],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )

    print("=== STDOUT ===")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    assert result.returncode == 0, (
        f"Smoke test exited with code {result.returncode}.\n"
        f"stdout tail:\n{result.stdout[-1000:]}\n"
        f"stderr:\n{result.stderr[:2000]}"
    )

    ckpt_dir = Path(CHECKPOINT_DIR)
    assert _find_checkpoint(ckpt_dir), (
        f"No checkpoint found under {ckpt_dir} after smoke run"
    )

    if ckpt_dir.is_dir():
        import shutil

        shutil.rmtree(ckpt_dir)
