# training/schedules.py
"""Batch-size and sequence-length schedules.

Both schedules interpolate from an initial value to a final value over
a fixed number of steps.  The default shape is linear, matching the
modded-nanogpt recipes (#46 for batch-size, #72 for seq-len).

Reference
---------
Keller Jordan, "modded-nanogpt speedrun", records #46 and #72 (April 2026).
"""

from __future__ import annotations


def _interpolate(
    step: int,
    initial: int,
    final: int,
    schedule_steps: int,
    shape: str = "linear",
) -> int:
    """Return the scheduled value at *step* for an ``initial → final`` ramp.

    Args:
        step: current training step (0-indexed).
        initial: starting value.
        final: ending value.
        schedule_steps: number of steps over which to ramp.
        shape: ``"linear"`` (default) or ``"step"`` (jump at midpoint).

    Returns:
        Scheduled value as an integer.
    """
    if step >= schedule_steps:
        return final
    if shape == "step":
        return final if step >= schedule_steps // 2 else initial
    progress = step / max(1, schedule_steps)
    return int(initial + (final - initial) * progress)


class BatchSizeSchedule:
    """Schedule micro-batch size from ``initial`` to ``final``.

    The schedule ramps ``micro_batch_size`` (and optionally
    ``gradient_accumulation_steps``) to increase total tokens per step
    as training progresses.  This matches modded-nanogpt #46.

    Note:
        Changing ``gradient_accumulation_steps`` mid-training affects
        the effective batch size but does **not** change the per-step
        loss normalisation.  The caller is responsible for updating the
        data loader's batch size and the gradient accumulation counter.
    """

    def __init__(
        self,
        initial_batch_size: int = 2,
        final_batch_size: int = 8,
        schedule_steps: int = 5_000,
        shape: str = "linear",
    ):
        assert 0 < initial_batch_size <= final_batch_size
        self.initial = initial_batch_size
        self.final = final_batch_size
        self.schedule_steps = schedule_steps
        self.shape = shape

    def get_batch_size(self, step: int) -> int:
        """Return the scheduled micro-batch size at *step*."""
        return _interpolate(step, self.initial, self.final, self.schedule_steps, self.shape)


class SeqLenSchedule:
    """Schedule max sequence length from ``initial`` to ``final``.

    This matches modded-nanogpt #72 — the model is initially trained on
    shorter sequences (faster, cheaper) and gradually lengthened to the
    target context size.

    Note:
        Changing ``max_seq_len`` mid-training requires the data loader
        to support variable-length sequences.  The schedule itself is
        stateless — it only computes the target value.  The caller must
        handle the actual re-packing of sequences.
    """

    def __init__(
        self,
        initial_seq_len: int = 2048,
        final_seq_len: int = 8192,
        schedule_steps: int = 5_000,
        shape: str = "linear",
    ):
        assert 0 < initial_seq_len <= final_seq_len
        self.initial = initial_seq_len
        self.final = final_seq_len
        self.schedule_steps = schedule_steps
        self.shape = shape

    def get_seq_len(self, step: int) -> int:
        """Return the scheduled sequence length at *step*."""
        return _interpolate(step, self.initial, self.final, self.schedule_steps, self.shape)
