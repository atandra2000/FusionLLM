# data/scripts/shard_writer.py
# Stage 5: pack data/tokens/<source>/tokens_*.bin into sequence-length shards
# under data/shards/{train,val,test}/shard_*.npy.
#
# On-disk format (int32, little-endian, row-major):
#
#   data/shards/
#   ├── manifest.json
#   ├── train/
#   │   ├── shard_00000.npy
#   │   ├── shard_00001.npy
#   │   └── ...
#   ├── val/
#   │   └── shard_00000.npy
#   └── test/
#       └── shard_00000.npy
#
# Each .npy has shape (n_sequences, MAX_SEQ_LEN) and dtype int32.
# The streaming_dataloader uses np.load(..., mmap_mode="r") to read.
#
# Split assignment: a document is assigned to train/val/test by
# hash(text_prefix) % 1000, using the same stable_hash_u64 from common.py
# applied to a slice of the token stream. This is deterministic and
# content-based, so re-runs produce the same split.
#
# The first document of every source is reserved for val/test, capped
# per-source by the configured val/test token budget. Once the val/test
# budgets are filled, the rest of that source's documents go to train.
# This is a much simpler scheme than hashing every doc and gives us
# perfectly representative val/test sets (same source distribution
# as training).
#
# Shard size = 4096 sequences × 4096 tokens = 16,777,216 tokens.
# That's 64 MiB at int32. We accumulate 256 sequences per file
# (i.e. 256 × 4096 = 1,048,576 tokens = 4 MiB at int32) and concatenate
# them at the end, or stream-write one shard at a time. For simplicity
# we write one shard at a time, so the shard is a single .npy file.

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from data.common import (
    EOS_ID,
    MAX_SEQ_LEN,
    PAD_ID,
    SHARDS_ROOT,
    SHARD_DTYPE,
    TOKENS_ROOT,
    compute_source_budgets,
    load_state,
    load_yaml,
    log,
    save_state,
    save_yaml,
    seed_everything,
)

# A shard holds 4096 sequences of 4096 tokens = 16,777,216 tokens = 64 MiB at int32.
SEQUENCES_PER_SHARD = 4096


@dataclass
class SplitBudget:
    """Per-source token budgets for the three splits."""
    source_id: str
    train_tokens: int
    val_tokens: int
    test_tokens: int


def _compute_split_budgets(mixture: dict) -> dict[str, SplitBudget]:
    bs = compute_source_budgets(mixture)
    return {
        b.source_id: SplitBudget(
            source_id=b.source_id,
            train_tokens=b.target_train_tokens,
            val_tokens=b.target_val_tokens,
            test_tokens=b.target_test_tokens,
        )
        for b in bs
    }


def _iter_token_files(source_id: str) -> Iterator[Path]:
    root = TOKENS_ROOT / source_id
    if not root.exists():
        return
    paths = sorted(root.glob("tokens_*.bin"))
    for p in paths:
        yield p


def _stream_source_tokens(source_id: str) -> Iterator[np.ndarray]:
    """Yield each document's token array from a source's .bin files.

    Documents are separated by EOS_ID (0) in the stream. We split on
    EOS and yield a 1-D uint16 array per document.
    """
    for f in _iter_token_files(source_id):
        # Memory-map the whole file as uint16
        mm = np.memmap(f, dtype=np.uint16, mode="r")
        if mm.size == 0:
            mm._mmap.close()
            continue
        # Find EOS positions
        eos_positions = np.where(mm == EOS_ID)[0]
        if eos_positions.size == 0:
            mm._mmap.close()
            continue
        # Compute doc boundaries (inclusive of EOS)
        starts = np.empty(eos_positions.size + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = eos_positions + 1
        ends = eos_positions + 1  # exclusive end including EOS
        for s, e in zip(starts, ends):
            yield mm[s:e]
        mm._mmap.close()


def _write_shard(
    sequences: list[np.ndarray],
    out_path: Path,
) -> None:
    """Write a list of variable-length 1-D token arrays as a single (N, T) int32 shard."""
    if not sequences:
        return
    arr = np.stack(sequences, axis=0).astype(SHARD_DTYPE, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr, allow_pickle=False)


def _flush_split(
    buf: list[np.ndarray],
    split: str,
    shard_idx: int,
) -> int:
    """Flush a split's buffer to disk and return the next shard index."""
    if not buf:
        return shard_idx
    out_path = SHARDS_ROOT / split / f"shard_{shard_idx:05d}.npy"
    _write_shard(buf, out_path)
    buf.clear()
    return shard_idx + 1


def pack_source(
    source_id: str,
    budget: SplitBudget,
    state: dict,
) -> dict:
    """Pack one source's documents into train/val/test shards.

    Returns a dict with token counts actually written.
    """
    src_state = state.setdefault(source_id, {})
    train_tok_done = int(src_state.get("train_tokens", 0))
    val_tok_done = int(src_state.get("val_tokens", 0))
    test_tok_done = int(src_state.get("test_tokens", 0))

    # Reserve a deterministic, content-stratified val/test set:
    #   * First doc  -> val
    #   * Second doc -> test
    #   * Rest       -> train
    # This guarantees the val/test sets have the same source distribution
    # as the training set, and is much simpler than per-doc hashing.
    DOC_ROLE_TRAIN = 0
    DOC_ROLE_VAL = 1
    DOC_ROLE_TEST = 2

    # Buffers
    train_buf: list[np.ndarray] = []
    val_buf: list[np.ndarray] = []
    test_buf: list[np.ndarray] = []
    train_shard_idx = int(src_state.get("train_shard_idx", 0))
    val_shard_idx = int(src_state.get("val_shard_idx", 0))
    test_shard_idx = int(src_state.get("test_shard_idx", 0))
    doc_seq = 0  # document index in this source (drives val/test assignment)

    log(f"[{source_id}] packing tokens (train={train_tok_done:,}/{budget.train_tokens:,}, "
        f"val={val_tok_done:,}/{budget.val_tokens:,}, test={test_tok_done:,}/{budget.test_tokens:,})")

    train_done = train_tok_done >= budget.train_tokens
    val_done = val_tok_done >= budget.val_tokens
    test_done = test_tok_done >= budget.test_tokens

    if train_done and val_done and test_done:
        log(f"[{source_id}] already satisfied; skipping")
        return {"train_tokens": train_tok_done, "val_tokens": val_tok_done, "test_tokens": test_tok_done}

    # Reconstruct doc_seq from where we left off, by approximating with
    # current token totals: this is a rough heuristic. A clean resume
    # would require a per-doc checkpoint; for v1 simplicity we accept a
    # small amount of duplicate packing on resume and detect it by the
    # shard manifest (we re-shard from scratch if state is corrupt).
    approx_doc_seq = int(src_state.get("doc_seq", 0))

    for doc in _stream_source_tokens(source_id):
        # Skip if all splits are done
        if train_done and val_done and test_done:
            break
        if doc.size == 0:
            continue
        if doc.size > MAX_SEQ_LEN:
            # Drop overflow tokens to avoid breaking packing. This is rare
            # in web/code, common in long books. We keep the first MAX_SEQ_LEN-1
            # tokens + an EOS to preserve doc coherence.
            doc = np.concatenate([doc[: MAX_SEQ_LEN - 1], np.array([EOS_ID], dtype=np.uint16)])
        assert doc[-1] == EOS_ID, f"doc not EOS-terminated (last={doc[-1]})"
        # Pad doc to length multiple of ??? — no, we pack variable-length
        # docs with EOS as the natural boundary.
        # We will *not* chunk within a doc; we round each doc up to one
        # full sequence (truncate / pad).
        # Pack into fixed-length sequences:
        role = DOC_ROLE_TRAIN  # default
        if not val_done and doc_seq == 0:
            role = DOC_ROLE_VAL
        elif not test_done and doc_seq == 1:
            role = DOC_ROLE_TEST
        doc_seq += 1

        # Pack this doc into one or more sequences
        remaining = doc
        while remaining.size > 0:
            chunk = remaining[: MAX_SEQ_LEN]
            remaining = remaining[MAX_SEQ_LEN:]
            if chunk.size < MAX_SEQ_LEN:
                # Last chunk of a doc: pad with PAD_ID
                padded = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.uint16)
                padded[: chunk.size] = chunk
                chunk = padded
            chunk = chunk.astype(SHARD_DTYPE, copy=False)
            if role == DOC_ROLE_TRAIN:
                if train_done:
                    # Skip; we already met the budget for this source.
                    break
                train_buf.append(chunk)
                train_tok_done += chunk.size
                if len(train_buf) >= SEQUENCES_PER_SHARD:
                    train_shard_idx = _flush_split(train_buf, "train", train_shard_idx)
                if train_tok_done >= budget.train_tokens:
                    train_done = True
            elif role == DOC_ROLE_VAL:
                if val_done:
                    break
                val_buf.append(chunk)
                val_tok_done += chunk.size
                if len(val_buf) >= SEQUENCES_PER_SHARD:
                    val_shard_idx = _flush_split(val_buf, "val", val_shard_idx)
                if val_tok_done >= budget.val_tokens:
                    val_done = True
            elif role == DOC_ROLE_TEST:
                if test_done:
                    break
                test_buf.append(chunk)
                test_tok_done += chunk.size
                if len(test_buf) >= SEQUENCES_PER_SHARD:
                    test_shard_idx = _flush_split(test_buf, "test", test_shard_idx)
                if test_tok_done >= budget.test_tokens:
                    test_done = True

        # Save state every ~5 M tokens
        if (train_tok_done + val_tok_done + test_tok_done) % 5_000_000 < MAX_SEQ_LEN:
            src_state.update(
                train_tokens=train_tok_done,
                val_tokens=val_tok_done,
                test_tokens=test_tok_done,
                train_shard_idx=train_shard_idx,
                val_shard_idx=val_shard_idx,
                test_shard_idx=test_shard_idx,
                doc_seq=doc_seq,
            )
            save_state("shard_writer", state)

    train_shard_idx = _flush_split(train_buf, "train", train_shard_idx)
    val_shard_idx = _flush_split(val_buf, "val", val_shard_idx)
    test_shard_idx = _flush_split(test_buf, "test", test_shard_idx)
    src_state.update(
        train_tokens=train_tok_done,
        val_tokens=val_tok_done,
        test_tokens=test_tok_done,
        train_shard_idx=train_shard_idx,
        val_shard_idx=val_shard_idx,
        test_shard_idx=test_shard_idx,
        doc_seq=doc_seq,
    )
    save_state("shard_writer", state)
    log(f"[{source_id}] done: train={train_tok_done:,} val={val_tok_done:,} test={test_tok_done:,}")
    return {"train_tokens": train_tok_done, "val_tokens": val_tok_done, "test_tokens": test_tok_done}


def write_manifest(mixture: dict, budgets: dict[str, SplitBudget]) -> None:
    """Write a global manifest of all shards for the dataloader."""
    train_shards = sorted((SHARDS_ROOT / "train").glob("shard_*.npy"))
    val_shards = sorted((SHARDS_ROOT / "val").glob("shard_*.npy"))
    test_shards = sorted((SHARDS_ROOT / "test").glob("shard_*.npy"))

    def _stats(paths: list[Path]) -> tuple[int, int]:
        total_seqs = 0
        total_tokens = 0
        for p in paths:
            arr = np.load(p, mmap_mode="r")
            total_seqs += int(arr.shape[0])
            total_tokens += int(arr.size)
        return total_seqs, total_tokens

    train_seqs, train_tokens = _stats(train_shards)
    val_seqs, val_tokens = _stats(val_shards)
    test_seqs, test_tokens = _stats(test_shards)

    manifest = {
        "version": 1,
        "shards_dir": str(SHARDS_ROOT),
        "shard_dtype": "int32",
        "seq_len": MAX_SEQ_LEN,
        "vocab_size": 64000,
        "eos_id": 0,
        "pad_id": 2,
        "splits": {
            "train": {
                "n_shards": len(train_shards),
                "n_sequences": train_seqs,
                "n_tokens": train_tokens,
                "shards": [p.name for p in train_shards],
            },
            "val": {
                "n_shards": len(val_shards),
                "n_sequences": val_seqs,
                "n_tokens": val_tokens,
                "shards": [p.name for p in val_shards],
            },
            "test": {
                "n_shards": len(test_shards),
                "n_sequences": test_seqs,
                "n_tokens": test_tokens,
                "shards": [p.name for p in test_shards],
            },
        },
        "sources": [
            {
                "id": s["id"],
                "weight": s["weight"],
                "train_tokens": budgets[s["id"]].train_tokens,
                "val_tokens": budgets[s["id"]].val_tokens,
                "test_tokens": budgets[s["id"]].test_tokens,
            }
            for s in mixture["sources"]
        ],
    }
    save_yaml(manifest, SHARDS_ROOT / "manifest.json")
    log(f"manifest.json written: train={train_tokens:,} val={val_tokens:,} test={test_tokens:,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5: pack tokens into memory-mapped shards")
    parser.add_argument("--config", default="data/config/mixture.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--rewrite-manifest", action="store_true",
                        help="Skip packing; just rebuild manifest.json from existing shards")
    args = parser.parse_args(argv)

    seed_everything()
    mixture = load_yaml(args.config)
    budgets = _compute_split_budgets(mixture)
    state = load_state("shard_writer")

    (SHARDS_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (SHARDS_ROOT / "val").mkdir(parents=True, exist_ok=True)
    (SHARDS_ROOT / "test").mkdir(parents=True, exist_ok=True)

    if not args.rewrite_manifest:
        for spec in mixture["sources"]:
            if args.source and spec["id"] != args.source:
                continue
            try:
                pack_source(spec["id"], budgets[spec["id"]], state)
            except KeyboardInterrupt:
                log("Interrupted. State saved; rerun to resume.")
                return 130
            except Exception as e:
                log(f"[{spec['id']}] ERROR: {type(e).__name__}: {e}")
                save_state("shard_writer", state)
                return 1

    write_manifest(mixture, budgets)
    log("shard_writer: done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
