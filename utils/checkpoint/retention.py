# utils/checkpoint/retention.py
"""Checkpoint retention and cleanup.

Handles listing, deleting, and pruning old checkpoints.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def list_steps(save_dir: Path) -> list[int]:
    """Return all step numbers that have checkpoint files or directories.
    
    Args:
        save_dir: Checkpoint directory
        
    Returns:
        List of step numbers
    """
    steps = set()
    # Sharded or DCP dirs: step_{n}/
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.add(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    # Flat safetensors: model_step_{n}.safetensors
    for p in save_dir.glob("model_step_*.safetensors"):
        try:
            steps.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return list(steps)


def checkpoint_complete(save_dir: Path, step: int) -> bool:
    """True iff all required files exist for this step.
    
    Args:
        save_dir: Checkpoint directory
        step: Step number to check
        
    Returns:
        True if checkpoint is complete
    """
    step_dir = save_dir / f"step_{step}"
    if step_dir.exists() and (step_dir / "meta.json").exists():
        return True
    return all(
        (save_dir / name).exists()
        for name in [
            f"model_step_{step}.safetensors",
            f"optim_step_{step}.pt",
            f"meta_step_{step}.json",
        ]
    )


def list_checkpoints(save_dir: Path) -> list[int]:
    """Return all complete checkpoint step numbers, sorted ascending.
    
    Args:
        save_dir: Checkpoint directory
        
    Returns:
        List of complete checkpoint steps
    """
    return sorted(s for s in list_steps(save_dir) if checkpoint_complete(save_dir, s))


def latest_step(save_dir: Path) -> int | None:
    """Return the highest complete step number, or None.
    
    Args:
        save_dir: Checkpoint directory
        
    Returns:
        Latest step or None
    """
    complete = list_checkpoints(save_dir)
    if not complete:
        return None
    return complete[-1]


def delete_checkpoint(save_dir: Path, step: int) -> None:
    """Remove all files for a given checkpoint step.
    
    Args:
        save_dir: Checkpoint directory
        step: Step to delete
    """
    step_dir = save_dir / f"step_{step}"
    if step_dir.exists():
        shutil.rmtree(step_dir)
        logger.info("[checkpoint] deleted step %d", step)
        return
    patterns = [
        f"model_step_{step}.safetensors",
        f"ema_step_{step}.safetensors",
        f"optim_step_{step}.pt",
        f"meta_step_{step}.json",
    ]
    for p in save_dir.glob(f"optim_*_step_{step}.pt"):
        patterns.append(p.name)
    for pattern in patterns:
        p = save_dir / pattern
        if p.exists():
            p.unlink()
    logger.info("[checkpoint] deleted step %d", step)


def keep_last_n(save_dir: Path, n: int) -> None:
    """Delete all but the `n` most recent complete checkpoints.
    
    best.safetensors / best_ema.safetensors are never deleted.
    
    Args:
        save_dir: Checkpoint directory
        n: Number of checkpoints to keep
    """
    complete = list_checkpoints(save_dir)
    for step in complete[:-n]:
        delete_checkpoint(save_dir, step)


__all__ = [
    "list_steps",
    "checkpoint_complete",
    "list_checkpoints",
    "latest_step",
    "delete_checkpoint",
    "keep_last_n",
]
