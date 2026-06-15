# training/checkpoint.py
"""Checkpoint save/load (safetensors format, BF16 weights)."""

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


def save_checkpoint(model: nn.Module, muon_opt: NorMuon | None, adamw_opt: CautiousAdamW | None, scheduler: WSDScheduler | None, step: int, token_count: int = 0, best_loss: float = float("inf"), save_dir: str = "checkpoints/pretrain", max_keep: int = 3, tag: str = "") -> Path:
    """Save training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"step_{step}"
    ckpt_dir = save_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_state = {k: v.contiguous().to(torch.bfloat16) for k, v in model.state_dict().items()}
    _save_safetensors(model_state, ckpt_dir / "model.safetensors")

    optim_state = {"muon": muon_opt.state_dict() if muon_opt else {}, "adamw": adamw_opt.state_dict() if adamw_opt else {}}
    torch.save(optim_state, ckpt_dir / "optimizer.pt")

    meta = {"step": step, "token_count": token_count, "best_loss": best_loss, "scheduler": scheduler.state_dict() if scheduler else {}, "model_config": getattr(model, "config", None)}
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(_make_serializable(meta), f, indent=2, default=str)

    print(f"[checkpoint] Saved at step {step} → {ckpt_dir}")
    _cleanup_old_checkpoints(save_dir, max_keep)
    return ckpt_dir


def load_checkpoint(model: nn.Module, muon_opt: NorMuon | None, adamw_opt: CautiousAdamW | None, scheduler: WSDScheduler | None, load_dir: str | Path, device: str = "cpu", strict: bool = True) -> dict[str, Any]:
    """Load training checkpoint."""
    load_dir = Path(load_dir)
    model_path = load_dir / "model.safetensors"
    if model_path.exists():
        model.load_state_dict(_load_safetensors(model_path, device=device), strict=strict)
        print(f"[checkpoint] Loaded model weights from {model_path}")

    optim_path = load_dir / "optimizer.pt"
    if optim_path.exists() and (muon_opt or adamw_opt):
        optim_state = torch.load(optim_path, map_location=device, weights_only=True)
        if muon_opt and "muon" in optim_state and optim_state["muon"]:
            muon_opt.load_state_dict(optim_state["muon"])
            print("[checkpoint] Loaded NorMuon state")
        if adamw_opt and "adamw" in optim_state and optim_state["adamw"]:
            adamw_opt.load_state_dict(optim_state["adamw"])
            print("[checkpoint] Loaded CautiousAdamW state")

    meta: dict[str, Any] = {}
    meta_path = load_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[checkpoint] Loaded metadata: step={meta.get('step', '?')}")
        if scheduler and "scheduler" in meta and meta["scheduler"]:
            scheduler.load_state_dict(meta["scheduler"])
            print("[checkpoint] Loaded scheduler state")
    return meta


def find_latest_checkpoint(save_dir: str | Path) -> Path | None:
    """Find latest checkpoint by step number."""
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
                    checkpoints.append((meta.get("step", -1), d))
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


def _load_safetensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
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
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _cleanup_old_checkpoints(save_dir: Path, max_keep: int) -> None:
    """Remove old checkpoints, keeping only max_keep."""
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
