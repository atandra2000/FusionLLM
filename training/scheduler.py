# training/scheduler.py
"""WSD (Warmup-Stable-Decay) Learning Rate Scheduler."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """Warmup-Stable-Decay scheduler."""

    def __init__(self, optimizer, total_steps: int = 63400, warmup_frac: float = 0.01, stable_frac: float = 0.84, min_lr_ratio: float = 0.1, decay: str = "linear", last_epoch: int = -1):
        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio
        assert decay in ("linear", "cosine"), f"decay must be 'linear' or 'cosine', got {decay!r}"
        self.decay = decay

        self.warmup_steps = int(total_steps * warmup_frac)
        self.stable_steps = int(total_steps * stable_frac)
        self.decay_steps = total_steps - self.warmup_steps - self.stable_steps

        self._optimizers = [optimizer] if not isinstance(optimizer, (list, tuple)) else list(optimizer)
        super().__init__(self._optimizers[0], last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.stable_steps:
            factor = 1.0
        elif step >= self.total_steps:
            factor = self.min_lr_ratio
        else:
            decay_progress = (step - self.warmup_steps - self.stable_steps) / max(1, self.decay_steps)
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress)) if self.decay == "cosine" else 1.0 - (1.0 - self.min_lr_ratio) * decay_progress
        return [base_lr * factor for base_lr in self.base_lrs]


class JointWSDScheduler:
    """WSD scheduler that drives multiple optimizers with one multiplicative factor.

    Ponytail: replaces per-optimizer _LRScheduler. The WSD curve is the same
    regardless of which optimizer; both NorMuon and AdamW get the same factor
    at every step, so the lr_muon/lr_adamw ratio is preserved across all
    phases. This is the fix for Bug B (WSD was only attached to AdamW).
    """

    def __init__(
        self,
        optimizers,
        total_steps: int = 63400,
        warmup_frac: float = 0.01,
        stable_frac: float = 0.84,
        min_lr_ratio: float = 0.1,
        decay: str = "linear",
    ):
        assert decay in ("linear", "cosine"), f"decay must be 'linear' or 'cosine', got {decay!r}"
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        self.optimizers = list(optimizers)

        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio
        self.decay = decay
        self.warmup_steps = int(total_steps * warmup_frac)
        self.stable_steps = int(total_steps * stable_frac)
        self.decay_steps = max(1, total_steps - self.warmup_steps - self.stable_steps)

        # Capture each param_group's base LR so we can scale by the WSD factor.
        self._base_lrs = [
            [g["lr"] for g in opt.param_groups]
            for opt in self.optimizers
        ]
        self.last_epoch = -1

    def _factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        if step < self.warmup_steps + self.stable_steps:
            return 1.0
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
        progress = max(0.0, min(1.0, progress))
        if self.decay == "cosine":
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self) -> None:
        self.last_epoch += 1
        f = self._factor(self.last_epoch)
        for opt, base_lrs in zip(self.optimizers, self._base_lrs):
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f

    def get_last_lr(self) -> list[float]:
        if not self.optimizers or not self.optimizers[0].param_groups:
            return [0.0]
        return [g["lr"] for g in self.optimizers[0].param_groups]
