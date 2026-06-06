# utils/checkpoint/atomic.py
"""Atomic file write helpers.

All file writes go through these functions to ensure atomicity
via temp-file + rename pattern.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def _json_default(obj):
    """JSON serialiser for types that json.dump cannot handle natively."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def atomic_save_safetensors(state: dict, path: Path, save_dir: Path) -> None:
    """Write a state dict as safetensors atomically via temp+rename.
    
    Args:
        state: State dict to save
        path: Final destination path
        save_dir: Directory for temp files
    """
    processed = {}
    for k, v in state.items():
        v = v.contiguous()
        if v.is_cuda:
            if v.is_pinned():
                v = v.to("cpu", non_blocking=True)
            else:
                v = v.pin_memory().to("cpu", non_blocking=True)
        processed[k] = v
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fd, tmp = tempfile.mkstemp(dir=save_dir, suffix=".safetensors.tmp")
    os.close(fd)
    try:
        save_file(processed, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_save_torch(obj, path: Path, save_dir: Path) -> None:
    """Pickle an object via torch.save atomically via temp+rename.
    
    Args:
        obj: Object to save
        path: Final destination path
        save_dir: Directory for temp files
    """
    fd, tmp = tempfile.mkstemp(dir=save_dir, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_save_json(obj: dict, path: Path, save_dir: Path) -> None:
    """Write a JSON file atomically via temp+rename.
    
    Args:
        obj: Dictionary to serialize
        path: Final destination path
        save_dir: Directory for temp files
    """
    fd, tmp = tempfile.mkstemp(dir=save_dir, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = [
    "atomic_save_safetensors",
    "atomic_save_torch",
    "atomic_save_json",
    "_json_default",
]
