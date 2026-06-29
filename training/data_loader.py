# training/data_loader.py
"""Async data loader with prefetching for training."""

from __future__ import annotations

import time
from typing import Iterator, Tuple

import torch


class AsyncDataLoader:
    """Async data loader with non-blocking GPU transfers and optional prefetching."""

    def __init__(
        self,
        data_iter: Iterator,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        prefetch_factor: int = 2,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.data_iter = data_iter
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self._prefetch_queue = None

    def _create_prefetch_iterator(self):
        """Create background prefetching."""
        def generator():
            while True:
                try:
                    tokens, targets = next(self.data_iter)
                    if self.pin_memory and tokens.device.type == 'cpu':
                        tokens = tokens.pin_memory()
                        targets = targets.pin_memory()
                    yield tokens, targets
                except StopIteration:
                    break

        return generator()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Return async iterator."""
        prefetch_iter = self._create_prefetch_iterator()

        for tokens, targets in prefetch_iter:
            tokens_gpu = tokens.to(self.device, non_blocking=True)
            targets_gpu = targets.to(self.device, non_blocking=True)
            yield tokens_gpu, targets_gpu

    def benchmark(self, num_batches: int = 100) -> dict:
        """Measure data loading throughput."""
        start = time.time()
        for i, (tokens, targets) in enumerate(self):
            if i >= num_batches:
                break

        elapsed = time.time() - start
        batches_per_sec = num_batches / elapsed
        tokens_loaded = num_batches * self.batch_size * self.seq_len

        return {
            'batches_per_sec': batches_per_sec,
            'tokens_per_sec': tokens_loaded / elapsed,
            'elapsed_sec': elapsed,
        }
