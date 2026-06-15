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

    def step_optimizers(self) -> None:
        """Step all optimizers."""
        self.step(self.last_epoch + 1)
        for opt in self._optimizers[1:]:
            lr = self.get_lr()[0]
            for pg in opt.param_groups:
                pg["lr"] = lr
