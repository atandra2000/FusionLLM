# training/curriculum_manager.py
"""Curriculum learning management.

Handles curriculum initialization and transitions during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data.curriculum import Curriculum

if TYPE_CHECKING:
    from training.configs import ConfigBundle


def init_curriculum(cfg: ConfigBundle) -> Curriculum | None:
    """Initialize curriculum learning if configured.

    Args:
        cfg: ConfigBundle with curriculum settings

    Returns:
        Curriculum instance or None
    """
    if cfg.curriculum_switch_step <= 0:
        return None
    if not cfg.data.shard_manifest_path:
        return None

    return Curriculum(
        manifest_path=Path(cfg.data.shard_manifest_path),
        stage_1_weights=cfg.curriculum_stage1_weights,
        stage_2_weights=cfg.curriculum_stage2_weights,
        switch_step=cfg.curriculum_switch_step,
        seed=0,
    )


def advance_curriculum(
    curriculum: Curriculum | None,
    step: int,
    loader,
    log_fn,
) -> None:
    """Advance curriculum if needed.

    Args:
        curriculum: Curriculum instance (or None)
        step: Current training step
        loader: Data loader (may have set_shards method)
        log_fn: Logging function
    """
    if curriculum is None:
        return

    if curriculum.advance(step):
        active_shards = curriculum.iter_active()
        log_fn(
            f"[curriculum] switched to {curriculum.active.name} "
            f"({len(active_shards)} shards in scope)"
        )
        if hasattr(loader, "set_shards"):
            loader.set_shards(active_shards)


__all__ = [
    "init_curriculum",
    "advance_curriculum",
]
