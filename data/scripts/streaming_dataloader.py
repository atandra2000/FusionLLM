# data/scripts/streaming_dataloader.py

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from data.common import (
    EOS_ID,
    MAX_SEQ_LEN,
    MASTER_SEED,
    SHARDS_ROOT,
    load_yaml,
    log,
    seed_everything,
)


def load_manifest(shards_dir: str | Path = SHARDS_ROOT) -> dict:
    p = Path(shards_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found at {p}")
    return load_yaml(p)


def _open_shards(split: str, manifest: dict) -> tuple[list[np.ndarray], int]:
    s = manifest["splits"][split]
    shard_names = s["shards"]
    split_dir = Path(manifest.get("shards_dir", str(SHARDS_ROOT))) / split
    shards: list[np.ndarray] = []
    for name in shard_names:
        p = split_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        assert arr.dtype == np.int32
        assert arr.ndim == 2 and arr.shape[1] == MAX_SEQ_LEN
        shards.append(arr)
    return shards, int(s["n_sequences"])


def make_dataloader(
    split: str = "train",
    *,
    micro_batch_size: int = 2,
    seq_len: int = MAX_SEQ_LEN,
    shuffle: bool = True,
    seed: int = MASTER_SEED,
    drop_last: bool = True,
    infinite: bool = True,
    shards_dir: str | Path = SHARDS_ROOT,
    device: str | torch.device | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if seq_len != MAX_SEQ_LEN:
        raise ValueError(f"seq_len={seq_len} != MAX_SEQ_LEN={MAX_SEQ_LEN}")

    manifest = load_manifest(shards_dir)
    shards, _ = _open_shards(split, manifest)
    if not shards:
        raise RuntimeError(f"No shards in split={split}")

    shard_sizes = [int(s.shape[0]) for s in shards]
    flat = []
    for si, n in enumerate(shard_sizes):
        flat.extend((si, qi) for qi in range(n))

    rng = random.Random(seed)
    device_t = torch.device(device) if device is not None else None

    def _epoch_pairs():
        order = list(range(len(flat)))
        if shuffle:
            rng.shuffle(order)
        return [flat[i] for i in order]

    while True:
        pairs = _epoch_pairs()
        for start in range(0, len(pairs), micro_batch_size):
            batch_pairs = pairs[start : start + micro_batch_size]
            if len(batch_pairs) < micro_batch_size and drop_last:
                break
            tokens_np = np.empty((micro_batch_size, seq_len), dtype=np.int32)
            for i, (si, qi) in enumerate(batch_pairs):
                tokens_np[i] = shards[si][qi]
            tokens = torch.from_numpy(tokens_np).long()
            targets = torch.roll(tokens, shifts=-1, dims=1)
            targets[:, -1] = EOS_ID
            if device_t is not None:
                tokens = tokens.to(device_t, non_blocking=True)
                targets = targets.to(device_t, non_blocking=True)
            yield tokens, targets
        if not infinite:
            return


def train_dataloader(
    *,
    micro_batch_size: int = 2,
    seq_len: int = MAX_SEQ_LEN,
    seed: int = MASTER_SEED,
    shards_dir: str | Path = SHARDS_ROOT,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    return make_dataloader(
        split="train",
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        shuffle=True,
        seed=seed,
        shards_dir=shards_dir,
    )


def val_dataloader(
    *,
    micro_batch_size: int = 2,
    seq_len: int = MAX_SEQ_LEN,
    seed: int = MASTER_SEED,
    shards_dir: str | Path = SHARDS_ROOT,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    return make_dataloader(
        split="val",
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        shuffle=False,
        seed=seed,
        infinite=False,
        shards_dir=shards_dir,
    )


def test_dataloader(
    *,
    micro_batch_size: int = 2,
    seq_len: int = MAX_SEQ_LEN,
    seed: int = MASTER_SEED,
    shards_dir: str | Path = SHARDS_ROOT,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    return make_dataloader(
        split="test",
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        shuffle=False,
        seed=seed,
        infinite=False,
        shards_dir=shards_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the streaming dataloader")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    seed_everything()
    log(f"Loading shards from {SHARDS_ROOT} (split={args.split})")
    manifest = load_manifest()
    n_seq = manifest["splits"][args.split]["n_sequences"]
    log(f"split={args.split} n_sequences={n_seq:,}")

    it = make_dataloader(
        split=args.split,
        micro_batch_size=args.micro_batch_size,
        shuffle=False,
        infinite=False,
    )
    for i, (tokens, targets) in enumerate(it):
        if i >= args.n_batches:
            break
        log(
            f"batch {i}: tokens.shape={tuple(tokens.shape)} dtype={tokens.dtype} "
            f"min={int(tokens.min())} max={int(tokens.max())} "
            f"targets.min={int(targets.min())} targets.max={int(targets.max())}"
        )
        assert int(targets[0, -1]) == EOS_ID
    log("dataloader smoke-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
