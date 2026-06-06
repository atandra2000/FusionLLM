# training/wsd.py
"""WSD (Warmup-Stable-Decay) learning-rate scheduler.

WSD divides training into three phases:

  1. **Warmup**   — LR increases linearly from 0 to peak over
                    ``warmup_frac × total_steps``.
  2. **Stable**   — LR stays at peak for
                    ``stable_frac × total_steps``.
  3. **Decay**    — LR decays linearly (or cosinely) from peak down to
                    ``min_lr_ratio × peak`` over
                    ``decay_frac × total_steps``.

Reference
---------
Hu et al., "WSD: Warmup-Stable-Decay Scheduling for Large-Scale Training",
arXiv:2503.09127, 2025.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """Warmup-Stable-Decay scheduler.

    Args:
        optimizer: wrapped optimizer(s).  May be a single optimizer
            or a list of optimizers.
        total_steps: total number of training steps.
        warmup_frac: fraction of total steps for linear warmup.
        stable_frac: fraction of total steps for constant peak LR.
        min_lr_ratio: LR decays to ``min_lr_ratio × peak``.
        decay: decay shape — ``"linear"`` or ``"cosine"``.
        last_epoch: the last epoch index (-1 for fresh start).
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
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

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.stable_steps:
            factor = 1.0
        elif step >= self.total_steps:
            factor = self.min_lr_ratio
        else:
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
