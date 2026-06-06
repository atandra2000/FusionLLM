# utils/tensor_checks.py
"""Shared tensor validation utilities for training stability.

Consolidates NaN/Inf checks that were previously duplicated across
training/pretrain.py, training/numerical_health.py, and other modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def validate_scalar(value: float, name: str, step: int = 0) -> None:
    """Check a scalar for NaN/Inf and raise RuntimeError if found.

    Used for loss values and other scalar metrics before they are used
    in backward passes or optimizer steps.
    """
    if math.isnan(value):
        raise RuntimeError(f"NaN {name} detected at step {step} — aborting training")
    if math.isinf(value):
        raise RuntimeError(f"Inf {name} detected at step {step} — aborting training")


def validate_tensor(tensor: torch.Tensor, name: str, step: int = 0) -> None:
    """Check a tensor for NaN/Inf and raise RuntimeError if found.

    Used for gradients, activations, and other tensor values.
    """
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN {name} detected at step {step} — aborting training")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"Inf {name} detected at step {step} — aborting training")


def validate_gradients(model: nn.Module, step: int = 0) -> None:
    """Check all gradients in a model for NaN/Inf.

    Called before the optimizer step to catch numerical issues early.
    """
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                raise RuntimeError(
                    f"NaN gradient detected in '{name}' at step {step} — aborting training"
                )
            if torch.isinf(p.grad).any():
                raise RuntimeError(
                    f"Inf gradient detected in '{name}' at step {step} — aborting training"
                )


def validate_loss(loss: torch.Tensor, step: int = 0) -> None:
    """Check a loss tensor for NaN/Inf before backward.

    Accepts a torch.Tensor (unlike validate_scalar which takes a float).
    """
    if torch.isnan(loss):
        raise RuntimeError(f"NaN loss detected at step {step} — aborting training")
    if torch.isinf(loss):
        raise RuntimeError(f"Inf loss detected at step {step} — aborting training")
