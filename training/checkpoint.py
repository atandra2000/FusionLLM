# training/checkpoint.py
"""Checkpoint save/load (Frozen v1 spec).

Checkpoint format (per FINAL_FROZEN_SPEC.md §8):
  - format: "safetensors"
  - precision: bf16 weights, fp32 optim
  - Contents:
    - model_state_dict (bf16, ~1.74 GB)
    - optimizer_state_dict (fp32, ~4.99 GB for active params)
    - scheduler_state
    - step
    - token_count
    - best_loss
  - save_interval_steps: 2000
  - save_max_keep: 3
  - synchronous save (async disabled)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.optimizer import NorMuon, CautiousAdamW
from training.scheduler import WSDScheduler


def save_checkpoint(
    model: nn.Module,
    muon_opt: NorMuon | None,
    adamw_opt: CautiousAdamW | None,
    scheduler: WSDScheduler | None,
    step: int,
    token_count: int = 0,
    best_loss: float = float("inf"),
    save_dir: str = "checkpoints/pretrain",
    max_keep: int = 3,
    tag: str = "",
) -> Path:
    """Save a training checkpoint.

    Args:
        model: The model (FusionLLM or MTP-wrapped).
        muon_opt: NorMuon optimizer.
        adamw_opt: CautiousAdamW optimizer.
        scheduler: WSD scheduler.
        step: Current training step.
        token_count: Total tokens processed.
        best_loss: Best validation loss so far.
        save_dir: Directory for checkpoints.
        max_keep: Maximum checkpoints to keep.
        tag: Optional tag (e.g., "final", "best").

    Returns:
        Path to the saved checkpoint directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = tag or f"step_{step}"
    ckpt_dir = save_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Save model weights (BF16) ─────────────────────────────────────────
    model_state = {k: v.contiguous().to(torch.bfloat16) for k, v in model.state_dict().items()}
    model_path = ckpt_dir / "model.safetensors"
    _save_safetensors(model_state, model_path)

    # ── Save optimizer states (FP32) ──────────────────────────────────────
    optim_state = {
        "muon": muon_opt.state_dict() if muon_opt is not None else {},
        "adamw": adamw_opt.state_dict() if adamw_opt is not None else {},
    }
    optim_path = ckpt_dir / "optimizer.pt"
    torch.save(optim_state, optim_path)

    # ── Save metadata ────────────────────────────────────────────────────
    meta = {
        "step": step,
        "token_count": token_count,
        "best_loss": best_loss,
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "model_config": getattr(model, "config", None),
    }
    meta_path = ckpt_dir / "metadata.json"
    with open(meta_path, "w") as f:
        # Convert non-serializable values
        meta_serializable = _make_serializable(meta)
        json.dump(meta_serializable, f, indent=2, default=str)

    print(f"[checkpoint] Saved at step {step} → {ckpt_dir}")

    # ── Cleanup old checkpoints ───────────────────────────────────────────
    _cleanup_old_checkpoints(save_dir, max_keep)

    return ckpt_dir


def load_checkpoint(
    model: nn.Module,
    muon_opt: NorMuon | None,
    adamw_opt: CautiousAdamW | None,
    scheduler: WSDScheduler | None,
    load_dir: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        model: Model to load weights into.
        muon_opt: NorMuon optimizer to restore state into.
        adamw_opt: CautiousAdamW optimizer to restore state into.
        scheduler: WSD scheduler to restore state into.
        load_dir: Path to checkpoint directory.
        device: Device to load onto.
        strict: Strict loading for model state dict.

    Returns:
        Metadata dict with step, token_count, best_loss, etc.
    """
    load_dir = Path(load_dir)

    # ── Load model weights ────────────────────────────────────────────────
    model_path = load_dir / "model.safetensors"
    if model_path.exists():
        model_state = _load_safetensors(model_path, device=device)
        model.load_state_dict(model_state, strict=strict)
        print(f"[checkpoint] Loaded model weights from {model_path}")
    else:
        print(f"[checkpoint] No model weights found at {model_path}")

    # ── Load optimizer states ─────────────────────────────────────────────
    optim_path = load_dir / "optimizer.pt"
    if optim_path.exists() and (muon_opt is not None or adamw_opt is not None):
        optim_state = torch.load(optim_path, map_location=device, weights_only=True)
        if muon_opt is not None and "muon" in optim_state and optim_state["muon"]:
            muon_opt.load_state_dict(optim_state["muon"])
            print("[checkpoint] Loaded NorMuon state")
        if adamw_opt is not None and "adamw" in optim_state and optim_state["adamw"]:
            adamw_opt.load_state_dict(optim_state["adamw"])
            print("[checkpoint] Loaded CautiousAdamW state")

    # ── Load metadata ─────────────────────────────────────────────────────
    meta_path = load_dir / "metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[checkpoint] Loaded metadata: step={meta.get('step', '?')}")

        # Restore scheduler state
        if scheduler is not None and "scheduler" in meta and meta["scheduler"]:
            scheduler.load_state_dict(meta["scheduler"])
            print("[checkpoint] Loaded scheduler state")

    return meta


def find_latest_checkpoint(save_dir: str | Path) -> Path | None:
    """Find the latest checkpoint directory by step number."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None

    checkpoints = []
    for d in save_dir.iterdir():
        if d.is_dir():
            try:
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    step = meta.get("step", -1)
                    checkpoints.append((step, d))
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def _save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    """Save state dict in safetensors format."""
    try:
        from safetensors.torch import save_file as st_save
        st_save(state_dict, str(path))
    except ImportError:
        print("[checkpoint] safetensors not available, falling back to torch.save")
        torch.save(state_dict, path.with_suffix(".pt"))


def _load_safetensors(
    path: Path, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors format."""
    try:
        from safetensors.torch import load_file as st_load
        return st_load(str(path), device=device)
    except ImportError:
        print("[checkpoint] safetensors not available, falling back to torch.load")
        return torch.load(path.with_suffix(".pt"), map_location=device, weights_only=True)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


def _cleanup_old_checkpoints(save_dir: Path, max_keep: int) -> None:
    """Remove old checkpoint directories, keeping only the most recent max_keep."""
    if max_keep <= 0:
        return

    checkpoints = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                checkpoints.append((step, d))
            except (ValueError, IndexError):
                continue

    checkpoints.sort(key=lambda x: x[0], reverse=True)

    for _, d in checkpoints[max_keep:]:
        shutil.rmtree(d)
        print(f"[checkpoint] Removed old checkpoint: {d}")
