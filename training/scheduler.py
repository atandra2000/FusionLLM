# training/scheduler.py
"""WSD (Warmup-Stable-Decay) Learning Rate Scheduler (Frozen v1 spec).

Schedule (per FINAL_FROZEN_SPEC.md §2):
  - warmup_frac = 0.01 (634 steps)
  - stable_frac = 0.84 (53,256 steps)
  - decay: linear from peak to min_lr_ratio × peak
  - min_lr_ratio = 0.1
  - total_steps = 63,400

WSD divides training into three phases:
  1. Warmup: LR increases linearly from 0 to peak.
  2. Stable: LR stays at peak.
  3. Decay: LR decays linearly to min_lr_ratio * peak.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """Warmup-Stable-Decay scheduler.

    Supports a single optimizer or a list of optimizers. When given
    a list, all optimizers are stepped together.

    Frozen v1 spec:
      - total_steps = 63,400
      - warmup_frac = 0.01 → 634 steps
      - stable_frac = 0.84 → 53,256 steps
      - decay: linear
      - min_lr_ratio = 0.1
    """

    def __init__(
        self,
        optimizer,
        total_steps: int = 63400,
        warmup_frac: float = 0.01,
        stable_frac: float = 0.84,
        min_lr_ratio: float = 0.1,
        decay: str = "linear",
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio
        assert decay in ("linear", "cosine"), f"decay must be 'linear' or 'cosine', got {decay!r}"
        self.decay = decay

        self.warmup_steps = int(total_steps * warmup_frac)
        self.stable_steps = int(total_steps * stable_frac)
        self.decay_steps = total_steps - self.warmup_steps - self.stable_steps

        # Wrap single optimizer in list for uniform handling
        if not isinstance(optimizer, (list, tuple)):
            optimizers = [optimizer]
        else:
            optimizers = list(optimizer)
        self._optimizers = optimizers

        super().__init__(optimizers[0], last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)

        if step < self.warmup_steps:
            # Linear warmup: 0 → 1
            factor = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.stable_steps:
            # Stable: 1.0
            factor = 1.0
        elif step >= self.total_steps:
            # Past end: min_lr_ratio
            factor = self.min_lr_ratio
        else:
            # Decay: peak → min_lr_ratio
            decay_progress = (step - self.warmup_steps - self.stable_steps) / max(
                1, self.decay_steps
            )
            if self.decay == "cosine":
                factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * decay_progress)
                )
            else:
                factor = 1.0 - (1.0 - self.min_lr_ratio) * decay_progress

        return [base_lr * factor for base_lr in self.base_lrs]

    def step_optimizers(self) -> None:
        """Step all optimizers (needed when using multiple optimizers)."""
        step_count = self.last_epoch + 1
        self.step(step_count)
        for opt in self._optimizers[1:]:
            lr = self.get_lr()[0]
            for pg in opt.param_groups:
                pg["lr"] = lr


class ConstantWarmupScheduler(_LRScheduler):
    """Simple constant LR after linear warmup (used as fallback)."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 634,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]
