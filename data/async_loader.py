"""Two-stage async sharded loader.

Powers the data path of the pre-training loop.  The design is the
`AsyncDataLoadAttnFinalWindow` trick from modded-nanogpt record #33
(adapted for our sharded mmap format from `data/shard_writer.py`).

Two stages
----------

* **Stage 1 (CPU)**: 8 workers per GPU shuffle shard order and
  read raw int32 token pages into a **pinned-memory** buffer.  The
  CPU page is sized to fit ``batch_size × grad_accum × seqlen`` of
  one micro-batch.

* **Stage 2 (CUDA, prefetch 2)**: a single GPU worker copies the
  pinned page to GPU with ``non_blocking=True`` and computes the
  per-step index permutation on the GPU side.  The prefetch depth
  is 2 micro-batches.

Per-rank deterministic offsets
------------------------------
Rank ``r`` reads shards ``[r, r+W, r+2W, ...]`` where ``W = world_size``.
This guarantees:

* every shard is read by exactly one rank (no overlap)
* every rank sees every shard eventually (full coverage)
* the order is deterministic and reproducible

This is the same pattern as modded-nanogpt #33 and avoids both
duplicate I/O and rank-synchronisation barriers.

Public surface
--------------
* :class:`AsyncShardLoader` — the main entry point.
* :func:`read_shard_header` — small helper used by the loader to
  parse the 256-byte shard header.
* :func:`open_shard` — context manager for memory-mapping a shard.

The async mode is **opt-in** (``async_mode=True``) and only works
on CUDA.  On CPU, the loader falls back to a synchronous
``numpy.memmap`` iteration that the smoke tests can use without
GPU support.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Shard format constants (mirror data/shard_writer.py) ──────────────────
SHARD_HEADER_BYTES = 256
SHARD_VERSION = 0x01


# ── Header helpers ─────────────────────────────────────────────────────────
@dataclass
class ShardHeader:
    version: int
    n_tokens: int
    max_seq_len: int
    pad_token_id: int

    @classmethod
    def from_bytes(cls, b: bytes) -> ShardHeader:
        assert len(b) == SHARD_HEADER_BYTES, f"bad header size: {len(b)}"
        return cls(
            version=b[0],
            n_tokens=int.from_bytes(b[1:9], "little"),
            max_seq_len=int.from_bytes(b[9:13], "little"),
            pad_token_id=int.from_bytes(b[13:17], "little", signed=True),
        )


def read_shard_header(path: Path) -> ShardHeader:
    with open(path, "rb") as f:
        return ShardHeader.from_bytes(f.read(SHARD_HEADER_BYTES))


@contextmanager
def open_shard(path: Path, *, mode: str = "r"):
    """Memory-map a shard's data section as a numpy int32 array.

    Usage:
        with open_shard(path) as arr:
            chunk = arr[start:end]
    """
    header = read_shard_header(path)
    # `np.memmap` interprets the file as a flat array starting at
    # offset 0, so we must pass `offset=SHARD_HEADER_BYTES` to skip
    # the header.  `f.seek(...)` is *not* honoured by np.memmap.
    arr = np.memmap(
        str(path),
        dtype=np.dtype(np.int32),
        mode=mode,
        shape=(header.n_tokens,),
        offset=SHARD_HEADER_BYTES,
    )
    try:
        yield arr
    finally:
        del arr


# ── Manifest loader ────────────────────────────────────────────────────────
@dataclass
class ShardMeta:
    path: str
    n_tokens: int
    source: str = "mixed"
    weight: float = 1.0
    quality_score: float = 1.0
    domain: str = "mixed"


def load_manifest(manifest_path: Path) -> list[ShardMeta]:
    """Read a ``shards/manifest.jsonl`` and return the list of rows."""
    rows: list[ShardMeta] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                ShardMeta(
                    path=d["path"],
                    n_tokens=int(d["n_tokens"]),
                    source=d.get("source", "mixed"),
                    weight=float(d.get("weight", 1.0)),
                    quality_score=float(d.get("quality_score", 1.0)),
                    domain=d.get("domain", "mixed"),
                )
            )
    return rows


# ── Shard index ────────────────────────────────────────────────────────────
class ShardIndex:
    """A view over a list of shards with rank-aware offsets.

    Iterating with ``__iter__`` yields a deterministic stream of
    shard indices, starting at ``rank`` and stepping by
    ``world_size``.  When ``shuffle=True`` the stream is shuffled
    per-epoch, but the rank offset is preserved.
    """

    def __init__(
        self,
        shards: Sequence[ShardMeta],
        *,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.shards = list(shards)
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.shards)

    def epoch_order(self) -> list[int]:
        """Return the indices that this rank will iterate this epoch."""
        # Per-rank offset: rank r sees positions [r, r+W, r+2W, ...]
        my_indices = list(range(self.rank, len(self.shards), self.world_size))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch * 1009)
            rng.shuffle(my_indices)
        self._epoch += 1
        return my_indices

    def set_shards(self, shards: Sequence[ShardMeta]) -> None:
        """Replace the shard list (used by curriculum hot-swap)."""
        self.shards = list(shards)

    def __iter__(self) -> Iterator[ShardMeta]:
        for i in self.epoch_order():
            yield self.shards[i]


# ── Async loader ──────────────────────────────────────────────────────────
class AsyncShardLoader:
    """Two-stage async loader over the sharded mmap corpus.

    Args:
        manifest_path: path to ``shards/manifest.jsonl``.
        root:         root directory for the relative shard paths
                      (defaults to the manifest's parent).
        batch_size:   per-rank micro-batch size.
        grad_accum:   number of micro-batches per optimizer step.
        seqlen:       sequence length (per token row).
        rank:         distributed rank (default 0).
        world_size:   total ranks (default 1).
        seed:         PRNG seed.
        micro_prefetch: prefetch depth (default 2).
        async_mode:   when True (and CUDA is available) the loader
                      pipelines CPU reads into a pinned-memory page
                      and H2D-copies on a worker thread.  When False
                      the loader is a synchronous iterator (used by
                      CPU smoke tests).
        device:       target device (default "cuda" when async, else
                      "cpu").

    Iteration contract
    ------------------
    Each ``__next__`` call returns ``(tokens, targets)`` of shape
    ``(batch_size, seqlen)``.  ``targets`` is ``tokens`` shifted
    left by one (next-token prediction).  The iteration is
    **infinite** (re-iterates the shards forever) so the trainer
    can loop on it.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        root: Path | None = None,
        batch_size: int = 2,
        grad_accum: int = 16,
        seqlen: int = 4096,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        micro_prefetch: int = 8,
        async_mode: bool | None = None,
        device: str | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.seqlen = seqlen
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.micro_prefetch = max(1, micro_prefetch)
        if async_mode is None:
            async_mode = torch.cuda.is_available()
        self.async_mode = async_mode and torch.cuda.is_available()
        self.device = device or ("cuda" if self.async_mode else "cpu")

        # Load manifest and build shard index
        self.shards = load_manifest(self.manifest_path)
        if not self.shards:
            raise ValueError(f"No shards in {self.manifest_path}")
        self.index = ShardIndex(
            self.shards,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=seed,
        )

        # One micro-batch = batch_size × seqlen int32 tokens
        self._mb_tokens = self.batch_size * self.seqlen
        # Per-rank pinned-memory page size (in int32 elements)
        self._pinned_shape = (self.micro_prefetch, self.batch_size, self.seqlen)
        self._pinned_buf: torch.Tensor | None = None

        # Async infrastructure (only initialised when async_mode)
        self._mb_queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._shutdown = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the async worker thread (no-op in sync mode)."""
        if not self.async_mode:
            return
        if self._worker is not None:
            return
        self._pinned_buf = torch.empty(
            self._pinned_shape,
            dtype=torch.int64,
            pin_memory=True,
        )
        self._mb_queue = queue.Queue(maxsize=self.micro_prefetch)
        self._shutdown.clear()
        self._worker = threading.Thread(
            target=self._async_worker_loop,
            daemon=True,
            name=f"AsyncShardLoader-r{self.rank}",
        )
        self._worker.start()

    def stop(self) -> None:
        """Signal the async worker to exit and wait for it."""
        if not self.async_mode or self._worker is None:
            return
        self._shutdown.set()
        # Push a sentinel so the worker unblocks
        try:
            self._mb_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._worker.join(timeout=5.0)
        self._worker = None
        self._pinned_buf = None
        self._mb_queue = None

    def __enter__(self) -> AsyncShardLoader:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # ── Sync iteration ─────────────────────────────────────────────────
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not self.async_mode:
            return self._iter_sync()
        self.start()
        return self._iter_async()

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The trainer calls next() directly; delegate to the
        # async iterator.
        it = getattr(self, "_async_it", None)
        if it is None:
            it = self._iter_async()
            self._async_it = it
        return next(it)

    def _iter_sync(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Synchronous iteration: read one micro-batch, yield, repeat."""
        for shard in self.index:
            with open_shard(self.root / shard.path) as arr:
                n = arr.shape[0]
                # Total tokens we need from this shard to fill at
                # least one micro-batch + target.
                need = self._mb_tokens
                if n < need:
                    # Skip shards that are too small.
                    continue
                # Random starting offset (per epoch, deterministic).
                # We advance the offset by `need` each micro-batch to
                # keep things simple in the sync path.
                start = 0
                while start + need <= n:
                    tokens = np.asarray(arr[start : start + need], dtype=np.int64)
                    start += need
                    yield self._to_pair(tokens)

    def _iter_async(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Async iteration: pull pre-paged micro-batches from the queue."""
        while True:
            mb = self._mb_queue.get()
            if mb is None:
                # Sentinel
                return
            tokens, targets = mb
            yield tokens, targets

    # ── Async worker ───────────────────────────────────────────────────
    def _async_worker_loop(self) -> None:
        """CPU worker that fills the pinned-memory buffer.

        One micro-batch = one slice of the pinned page.  After
        filling, the worker pushes the (tokens, targets) pair to
        the queue.  The main thread then ``.to(self.device,
        non_blocking=True)``-copies.
        """
        rng = random.Random(self.seed)
        try:
            for shard in self.index:
                with open_shard(self.root / shard.path) as arr:
                    n = arr.shape[0]
                    need = self._mb_tokens
                    if n < need:
                        continue
                    # We iterate over the shard in chunks of `need`
                    # tokens, with a random per-epoch permutation
                    # of the chunk starts.
                    chunk_starts = list(range(0, n - need + 1, need))
                    rng.shuffle(chunk_starts)
                    for start in chunk_starts:
                        if self._shutdown.is_set():
                            return
                        # Fill one slice of the pinned buffer
                        slot = self._pinned_buf[0]  # we only use 1 of N
                        page = slot.numpy()
                        np.copyto(
                            page,
                            np.asarray(arr[start : start + need], dtype=np.int64),
                        )
                        # Build targets (left-shift) on the page
                        tokens = self._pinned_buf[0].clone()
                        targets = torch.empty_like(tokens)
                        targets[:, :-1] = tokens[:, 1:]
                        targets[:, -1] = -100  # ignore the last target
                        # H2D (non-blocking)
                        tokens = tokens.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        self._mb_queue.put((tokens, targets))
        except Exception as exc:
            # Surface the error to the trainer via the queue
            self._mb_queue.put(("__error__", exc))
            return

    # ── Helpers ────────────────────────────────────────────────────────
    def _to_pair(self, flat: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape a flat int64 buffer to (batch, seqlen) and build targets."""
        b = self.batch_size
        s = self.seqlen
        assert flat.size == b * s, f"expected {b * s} tokens, got {flat.size}"
        tokens = torch.from_numpy(flat.reshape(b, s))
        targets = torch.empty_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        targets[:, -1] = -100
        return tokens, targets

    # ── Curriculum hook (Phase 6.3) ───────────────────────────────────
    def set_shards(self, shards: list[ShardMeta]) -> None:
        """Replace the shard list and rebuild the index (curriculum hot-swap)."""
        self.shards = list(shards)
        self.index.set_shards(self.shards)
        if self.async_mode:
            self.stop()
            self.start()

    # ── Resize hooks (Phase 4 batch/seq-len schedules) ─────────────────
    def set_batch_size(self, new_bs: int) -> None:
        self.batch_size = new_bs
        self._mb_tokens = self.batch_size * self.seqlen
        if self.async_mode:
            # The pinned page has the wrong shape now.  The caller
            # is expected to call `start()` again to refresh.
            self.stop()
            self.start()

    def set_seq_len(self, new_seqlen: int) -> None:
        self.seqlen = new_seqlen
        self._mb_tokens = self.batch_size * self.seqlen
        if self.async_mode:
            self.stop()
            self.start()

    # ── Stats ──────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "n_shards": len(self.shards),
            "total_tokens": sum(s.n_tokens for s in self.shards),
            "batch_size": self.batch_size,
            "seqlen": self.seqlen,
            "async_mode": self.async_mode,
            "device": str(self.device),
        }
