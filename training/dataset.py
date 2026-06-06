# training/dataset.py
"""Packed pre-training dataset backed by a flat token tensor.

Each sample is ``(inp, tgt)`` with ``inp = tokens[i:i+L]`` and
``tgt = tokens[i+1:i+L+1]``. If the on-disk file is missing a
synthetic random dataset is generated and persisted so training can
proceed.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Packed pre-training dataset."""

    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        if os.path.exists(data_path):
            self.data = torch.load(data_path, weights_only=True)
        else:
            print(f"[warn] {data_path} not found — generating synthetic data")
            os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
            self.data = torch.randint(0, vocab_size, (1_000_000,))
            torch.save(self.data, data_path)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.max_seq_len

    def __getitem__(self, idx: int):
        start = idx * self.max_seq_len
        chunk = self.data[start : start + self.max_seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


__all__ = ["PretrainDataset"]
