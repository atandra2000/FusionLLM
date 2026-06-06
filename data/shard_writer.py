# data/shard_writer.py
"""Webdataset-style sharded mmap writer (Phase 1.3).

Writes token arrays to 4 GB-token shards with a 256-byte header and
atomic-rename semantics.  Produces ``shards/manifest.jsonl`` with one
row per shard ``{path, n_tokens, source, weight, quality_score, domain}``.

Usage
-----
    writer = ShardWriter(output_dir="data", target_shard_tokens=4_000_000_000)
    manifest = writer.write(tokens, max_seq_len=8192, pad_token_id=0)
    ShardWriter.write_manifest(manifest, output_dir)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

TARGET_SHARD_TOKENS = 4_000_000_000
HEADER_VERSION = 0x01
HEADER_SIZE = 256


def np_dtype():
    import numpy as np

    return np.int32


def _atomic_write_shard(path: Path, tokens, max_seq_len: int, pad_token_id: int) -> None:
    """Write a single shard atomically (temp + rename)."""
    header = bytearray(HEADER_SIZE)
    header[0] = HEADER_VERSION
    header[1:9] = len(tokens).to_bytes(8, "little")
    header[9:13] = max_seq_len.to_bytes(4, "little")
    pid = pad_token_id if pad_token_id is not None else -1
    header[13:17] = pid.to_bytes(4, "little", signed=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".bin.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(header)
            f.write(tokens.tobytes())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_shards(
    tokens,
    output_dir: Path,
    *,
    target_shard_tokens: int = TARGET_SHARD_TOKENS,
    max_seq_len: int = 8192,
    pad_token_id: int = -1,
) -> list[dict]:
    """Write token array to webdataset-style shards.

    Each shard is a 256-byte header + ``n_tokens × 4`` bytes (int32).

    Returns a list of manifest rows (one per shard).  The caller
    writes them to ``shards/manifest.jsonl``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    n_tokens = tokens.numel()
    n_shards = max(1, (n_tokens + target_shard_tokens - 1) // target_shard_tokens)
    print(
        f"\n[shards] writing {n_tokens:,} tokens \u2192 {n_shards} shard(s) of "
        f"~{target_shard_tokens:,} tokens each at {shards_dir}"
    )

    manifest: list[dict] = []
    tokens_np = tokens.numpy().astype(np_dtype())

    for idx in range(n_shards):
        start = idx * target_shard_tokens
        end = min(start + target_shard_tokens, n_tokens)
        shard_tokens = tokens_np[start:end]
        shard_path = shards_dir / f"shard_{idx:05d}.bin"
        _atomic_write_shard(shard_path, shard_tokens, max_seq_len, pad_token_id)
        size_gb = shard_path.stat().st_size / (1024**3)
        print(
            f"    shard {idx}: {shard_path.name} ({len(shard_tokens):,} tokens, {size_gb:.2f} GB)"
        )
        manifest.append(
            {
                "path": f"shards/{shard_path.name}",
                "n_tokens": int(len(shard_tokens)),
                "source": "mixed",
                "weight": 1.0,
                "quality_score": 1.0,
                "domain": "mixed",
            }
        )

    return manifest


def write_manifest(manifest: list[dict], output_dir: Path) -> Path:
    """Write ``shards/manifest.jsonl`` and return the path."""
    path = output_dir / "shards" / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"    manifest \u2192 {path} ({len(manifest)} rows)")
    return path


__all__ = ["write_shards", "write_manifest", "np_dtype", "TARGET_SHARD_TOKENS"]
