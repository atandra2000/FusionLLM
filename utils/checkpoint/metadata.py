# utils/checkpoint/metadata.py
"""Checkpoint metadata handling.

Manages building, loading, and updating checkpoint metadata
including best validation loss tracking.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from utils.checkpoint.atomic import atomic_save_json

logger = logging.getLogger(__name__)


def build_meta(
    step: int,
    best_val_loss: float,
    val_loss: float | None = None,
    extra_meta: dict | None = None,
) -> dict:
    """Build the metadata dict shared by all save paths.
    
    Args:
        step: Training step
        best_val_loss: Best validation loss seen so far
        val_loss: Optional current validation loss
        extra_meta: Optional extra metadata
        
    Returns:
        Metadata dictionary
    """
    meta: dict = {"step": step, "best_val_loss": best_val_loss}
    if val_loss is not None:
        meta["val_loss"] = val_loss
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if k != "step"})
    return meta


def load_best_val_loss(save_dir: Path) -> float:
    """Restore best_val_loss from best_meta.json if it exists.
    
    Args:
        save_dir: Checkpoint directory
        
    Returns:
        Best validation loss (inf if not found)
    """
    best_meta_path = save_dir / "best_meta.json"
    if best_meta_path.exists():
        try:
            with open(best_meta_path) as f:
                d = json.load(f)
            best_val_loss = float(d.get("best_val_loss", float("inf")))
            logger.info(
                "[checkpoint] restored best_val_loss=%.4f from %s",
                best_val_loss,
                best_meta_path,
            )
            return best_val_loss
        except Exception as exc:
            logger.warning("[checkpoint] could not read best_meta.json: %s", exc)
    return float("inf")


def maybe_update_best(
    save_dir: Path,
    state: dict,
    ema_state: dict | None,
    step: int,
    val_loss: float | None,
    best_val_loss: float,
    best_lock: threading.Lock,
) -> float:
    """Update best checkpoint if val_loss improved (thread-safe).
    
    Args:
        save_dir: Checkpoint directory
        state: Model state dict
        ema_state: Optional EMA state dict
        step: Current step
        val_loss: Current validation loss
        best_val_loss: Current best validation loss
        best_lock: Thread lock for best_val_loss
        
    Returns:
        Updated best_val_loss
    """
    with best_lock:
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            _update_best(save_dir, state, ema_state, step, val_loss)
    return best_val_loss


def _update_best(
    save_dir: Path,
    state: dict,
    ema_state: dict | None,
    step: int,
    val_loss: float,
) -> None:
    """Copy current weights to best.safetensors (and best_ema.safetensors).
    
    Args:
        save_dir: Checkpoint directory
        state: Model state dict
        ema_state: Optional EMA state dict
        step: Current step
        val_loss: Validation loss
    """
    from utils.checkpoint.atomic import atomic_save_safetensors
    
    best_path = save_dir / "best.safetensors"
    atomic_save_safetensors(state, best_path, save_dir)

    if ema_state is not None:
        best_ema_path = save_dir / "best_ema.safetensors"
        atomic_save_safetensors(ema_state, best_ema_path, save_dir)

    # Persist best metadata so best_val_loss survives restarts
    best_meta = {"step": step, "best_val_loss": val_loss}
    atomic_save_json(best_meta, save_dir / "best_meta.json", save_dir)

    logger.info(
        "[checkpoint] new best val_loss=%.4f at step %d → best.safetensors",
        val_loss,
        step,
    )


__all__ = [
    "build_meta",
    "load_best_val_loss",
    "maybe_update_best",
]
