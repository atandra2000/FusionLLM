# training/data_loader.py
"""Data loading: move batches to device with non-blocking transfers."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch


# ponytail: replaces AsyncDataLoader — prefetch_factor/num_workers/pin_memory
# were advertised but never read; the "prefetch iterator" was a plain sync
# generator. A single generator function is the whole contract.
def to_device_batches(
    data_iter: Iterator, device: torch.device
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield (tokens, targets) moved to ``device`` with non_blocking transfer."""
    for tokens, targets in data_iter:
        yield tokens.to(device, non_blocking=True), targets.to(device, non_blocking=True)