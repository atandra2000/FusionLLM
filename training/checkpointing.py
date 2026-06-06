# training/checkpointing.py
"""Checkpoint save/load orchestration.

Handles the high-level checkpoint operations for the training loop,
including metadata construction and state restoration.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    import torch
    from training.configs import ConfigBundle
    from training.optimization import Muon


def save_checkpoint(
    step: int,
    tag: str,
    cfg: ConfigBundle,
    ckpt_manager: CheckpointManager,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    adamw: torch.optim.Optimizer,
    muon: Muon | None,
    scheduler,
    curriculum,
    opt_steps: int,
    world_size: int,
    rank: int,
    log_fn,
) -> None:
    """Save a checkpoint.

    Args:
        step: Current training step
        tag: Checkpoint tag
        cfg: ConfigBundle
        ckpt_manager: CheckpointManager instance
        model: Wrapped model (possibly FSDP)
        raw_model: Unwrapped model for state_dict
        adamw: AdamW optimizer
        muon: Optional Muon optimizer
        scheduler: LR scheduler
        curriculum: Optional Curriculum instance
        opt_steps: Number of optimizer steps taken
        world_size: Number of distributed ranks
        rank: This rank's index
        log_fn: Logging function
    """
    extra_meta = {
        "scheduler": scheduler.state_dict(),
        "opt_steps": opt_steps,
        "tag": tag or f"step_{step}",
        "config": asdict(cfg),
    }
    # Save curriculum state for restart recovery
    if curriculum is not None:
        extra_meta["curriculum_advanced"] = curriculum._advanced
        extra_meta["curriculum_switch_step"] = curriculum.switch_step

    extra_optimizers = {}
    if muon is not None:
        extra_optimizers["muon"] = muon

    if cfg.checkpoint.checkpoint_backend == "dcp" and world_size > 1:
        ckpt_manager.save_fsdp2_dcp(
            model,
            adamw,
            step,
            extra_meta=extra_meta,
            keep_last_n=cfg.shard_keep_last if hasattr(cfg, 'shard_keep_last') else 5,
        )
    else:
        if rank != 0:
            return
        model_to_save = raw_model
        ckpt_manager.save(
            model_to_save, adamw, step,
            extra_meta=extra_meta,
            extra_optimizers=extra_optimizers if extra_optimizers else None,
            keep_last_n=5,
        )
    log_fn(f"Checkpoint saved at step {step}")


def load_checkpoint(
    step: int,
    cfg: ConfigBundle,
    ckpt_manager: CheckpointManager,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    adamw: torch.optim.Optimizer,
    muon: Muon | None,
    scheduler,
    curriculum,
    device: str,
    log_fn,
) -> tuple[int, int]:
    """Load a checkpoint.

    Args:
        step: Checkpoint step to load
        cfg: ConfigBundle
        ckpt_manager: CheckpointManager instance
        model: Wrapped model
        raw_model: Unwrapped model
        adamw: AdamW optimizer
        muon: Optional Muon optimizer
        scheduler: LR scheduler
        curriculum: Optional Curriculum instance
        device: Device string
        log_fn: Logging function

    Returns:
        (resumed_step, opt_steps)
    """
    extra_optimizers = {}
    if muon is not None:
        extra_optimizers["muon"] = muon

    if cfg.checkpoint.checkpoint_backend == "dcp" and cfg.data.gradient_accumulation_steps > 1:
        meta = ckpt_manager.load_fsdp2_dcp(
            model,
            adamw,
            step,
        )
    else:
        meta = ckpt_manager.load(
            raw_model,
            step,
            device=device,
            optimizer=adamw,
            extra_optimizers=extra_optimizers if extra_optimizers else None,
        )
    if "scheduler" in meta:
        scheduler.load_state_dict(meta["scheduler"])
    
    opt_steps = meta.get("opt_steps", 0)

    # Restore curriculum state
    if curriculum is not None and "curriculum_advanced" in meta:
        curriculum._advanced = meta["curriculum_advanced"]
        if curriculum._advanced:
            curriculum._active = curriculum.stage_2
            log_fn(f"[curriculum] restored advanced=True (stage 2 active)")

    # Verify NorMuon state restored correctly
    if muon is not None and extra_optimizers:
        _verify_nor_muon_state(muon, step, log_fn)

    resumed_step = meta.get("step", step)
    log_fn(f"Resumed from step {resumed_step}")
    return resumed_step, opt_steps


def _verify_nor_muon_state(muon, step: int, log_fn) -> None:
    """Verify NorMuon optimizer state was restored correctly."""
    if muon is None:
        return
    n_with_state = 0
    for p in muon.params:
        if p in muon.state and len(muon.state[p]) > 0:
            n_with_state += 1
    if n_with_state == 0:
        log_fn(f"[warn] NorMuon state empty after loading step {step} — optimizer may have lost momentum")
    else:
        log_fn(f"[ckpt] NorMuon state verified: {n_with_state} tensors with state")


def find_latest_checkpoint(ckpt_manager: CheckpointManager) -> int | None:
    """Find the latest complete checkpoint step."""
    return ckpt_manager.latest_step()


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]
