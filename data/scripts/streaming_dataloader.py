# data/scripts/streaming_dataloader.py
# Stage 6: stream memory-mapped shards and yield (tokens, targets) batches
# in the exact shape the Trainer expects: (micro_batch_size, seq_len) long.
#
# targets are computed on-the-fly with torch.roll (shift by 1) and the
# last position filled with EOS. This matches what trainer.py does
# internally (see trainer.py:213-215) and is cheaper than storing a
# separate target file.
#
# MTP heads consume `tokens` directly (see models/mtp.py:197-198) so the
# rolled targets we provide here are the *main* loss targets; MTP builds
# its own shifted targets. No special handling needed.
#
# Usage:
#
#   from data.scripts.streaming_dataloader import make_dataloader
#   data_iter = make_dataloader(split="train", micro_batch_size=2, seq_len=4096)
#   trainer.train_epoch(data_iter)

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
    """Load the global shard manifest written by shard_writer.py."""
    p = Path(shards_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {p}. Run shard_writer.py first."
        )
    return load_yaml(p)


def _open_shards(split: str, manifest: dict) -> tuple[list[np.ndarray], int]:
    """Memory-map every shard in a split. Returns (shards, total_sequences)."""
    s = manifest["splits"][split]
    shard_names = s["shards"]
    split_dir = Path(manifest.get("shards_dir", str(SHARDS_ROOT))) / split
    shards: list[np.ndarray] = []
    for name in shard_names:
        p = split_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        assert arr.dtype == np.int32, f"shard {p} has dtype {arr.dtype}, expected int32"
        assert arr.ndim == 2 and arr.shape[1] == MAX_SEQ_LEN, (
            f"shard {p} has shape {arr.shape}, expected (N, {MAX_SEQ_LEN})"
        )
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
    """Yield (tokens, targets) batches of shape (micro_batch_size, seq_len) long.

    Args:
        split:       "train" | "val" | "test"
        micro_batch_size: matches Trainer.config["micro_batch_size"]
        seq_len:     sequence length (default 4096 to match the architecture)
        shuffle:     shuffle the order of (shard, sequence) pairs
        seed:        RNG seed for shuffling
        drop_last:   drop the final partial batch (the trainer also does this)
        infinite:    if True, cycle forever (the trainer expects this)
        shards_dir:  path to data/shards
        device:      if set, move the returned tensors to this device

    Yields:
        (tokens, targets), each (micro_batch_size, seq_len) torch.long
    """
    if seq_len != MAX_SEQ_LEN:
        # We could support variable seq_len by slicing, but the architecture
        # is frozen at 4096. Refuse loudly rather than silently misalign.
        raise ValueError(
            f"seq_len={seq_len} != MAX_SEQ_LEN={MAX_SEQ_LEN}; the architecture is frozen at 4096."
        )

    manifest = load_manifest(shards_dir)
    shards, _ = _open_shards(split, manifest)
    if not shards:
        raise RuntimeError(f"No shards in split={split}; cannot stream.")

    # Per-shard sequence counts
    shard_sizes = [int(s.shape[0]) for s in shards]
    # Build a flat index of (shard_idx, seq_idx)
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
        # Iterate in micro-batches
        for start in range(0, len(pairs), micro_batch_size):
            batch_pairs = pairs[start : start + micro_batch_size]
            if len(batch_pairs) < micro_batch_size and drop_last:
                break
            # Materialize the batch: read micro_batch_size sequences from
            # the appropriate shards. This is the only allocation; the
            # underlying shards are mmapped, so RSS stays small.
            tokens_np = np.empty((micro_batch_size, seq_len), dtype=np.int32)
            for i, (si, qi) in enumerate(batch_pairs):
                tokens_np[i] = shards[si][qi]
            tokens = torch.from_numpy(tokens_np).long()
            # targets[t] = tokens[t+1] (rolled), with EOS at the last position.
            # This is the standard next-token prediction target and matches
            # the trainer's internal computation (see trainer.py).
            targets = torch.roll(tokens, shifts=-1, dims=1)
            targets[:, -1] = EOS_ID
            if device_t is not None:
                tokens = tokens.to(device_t, non_blocking=True)
                targets = targets.to(device_t, non_blocking=True)
            yield tokens, targets
        if not infinite:
            return


# ── Convenience wrappers matching the trainer's expected interface ───────
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
        infinite=False,  # validation is a single pass
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


# ── CLI: smoke-test the dataloader on the existing shards ───────────────
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
        # Cross-check: targets[i, t] should equal tokens[i, t+1] except at the last column
        # where targets[i, -1] == EOS_ID
        assert int(targets[0, -1]) == EOS_ID, f"last col of targets should be EOS={EOS_ID}, got {int(targets[0,-1])}"
    log("dataloader smoke-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
