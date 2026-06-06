"""Unit tests for `data/async_loader.py`.

Phase 1.5 scope:
  * Shard header round-trip (256 bytes, fields parse correctly).
  * Manifest loader (jsonl → ShardMeta list).
  * ShardIndex: per-rank deterministic offsets, no overlap, full
    coverage.
  * AsyncShardLoader sync path (CPU): iteration shape, targets
    left-shift, infinite loop.
  * set_batch_size / set_seq_len resize hooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data.async_loader import (
    AsyncShardLoader,
    ShardHeader,
    ShardIndex,
    ShardMeta,
    load_manifest,
    open_shard,
    read_shard_header,
)
from data.prepare_data import write_manifest, write_shards


# ── ShardHeader ────────────────────────────────────────────────────────────
class TestShardHeader:
    def test_round_trip(self, tmp_path: Path):
        # Write a small shard via the writer (which also writes a header)
        tokens = torch.arange(50, dtype=torch.long)
        write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=100,
            max_seq_len=10,
            pad_token_id=-1,
        )
        h = read_shard_header(tmp_path / "shards" / "shard_00000.bin")
        assert h.version == 0x01
        assert h.n_tokens == 50
        assert h.max_seq_len == 10
        assert h.pad_token_id == -1

    def test_pad_token_id_positive(self, tmp_path: Path):
        tokens = torch.arange(10, dtype=torch.long)
        write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=100,
            max_seq_len=5,
            pad_token_id=42,
        )
        h = read_shard_header(tmp_path / "shards" / "shard_00000.bin")
        assert h.pad_token_id == 42


# ── open_shard (mmap context manager) ─────────────────────────────────────
class TestOpenShard:
    def test_open_and_read(self, tmp_path: Path):
        tokens = torch.arange(100, dtype=torch.long)
        write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=100,
            max_seq_len=10,
            pad_token_id=0,
        )
        with open_shard(tmp_path / "shards" / "shard_00000.bin") as arr:
            assert arr.shape == (100,)
            assert arr.dtype == np.int32
            assert arr[0] == 0
            assert arr[99] == 99


# ── Manifest loader ───────────────────────────────────────────────────────
class TestLoadManifest:
    def test_round_trip(self, tmp_path: Path):
        tokens = torch.arange(100, dtype=torch.long)
        manifest = write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=50,
            max_seq_len=10,
            pad_token_id=0,
        )
        path = write_manifest(manifest, tmp_path)
        rows = load_manifest(path)
        assert len(rows) == 2
        assert [r.n_tokens for r in rows] == [50, 50]
        # Manifests are written with default fields (source/weight/etc)
        for r in rows:
            assert r.source == "mixed"
            assert r.weight == 1.0


# ── ShardIndex ────────────────────────────────────────────────────────────
class TestShardIndex:
    def _rows(self, n: int) -> list:
        return [ShardMeta(path=f"shard_{i:05d}.bin", n_tokens=100) for i in range(n)]

    def test_no_overlap_full_coverage(self):
        # Two ranks, 7 shards: each rank sees 3-4 shards, no overlap.
        idx0 = ShardIndex(self._rows(7), rank=0, world_size=2, shuffle=False, seed=0)
        idx1 = ShardIndex(self._rows(7), rank=1, world_size=2, shuffle=False, seed=0)
        seen0 = [s.path for s in idx0]
        seen1 = [s.path for s in idx1]
        # No overlap
        assert not (set(seen0) & set(seen1))
        # Full coverage
        assert sorted(seen0 + seen1) == sorted([f"shard_{i:05d}.bin" for i in range(7)])

    def test_shuffle_is_deterministic(self):
        idx_a = ShardIndex(self._rows(10), rank=0, world_size=1, shuffle=True, seed=42)
        idx_b = ShardIndex(self._rows(10), rank=0, world_size=1, shuffle=True, seed=42)
        order_a = [s.path for s in idx_a]
        order_b = [s.path for s in idx_b]
        assert order_a == order_b

    def test_shuffle_differs_across_epochs(self):
        idx = ShardIndex(self._rows(20), rank=0, world_size=1, shuffle=True, seed=0)
        first_epoch = [s.path for s in idx]
        second_epoch = [s.path for s in idx]
        # With 20 elements and a seeded shuffle, two consecutive
        # epochs are extremely unlikely to be identical.
        assert first_epoch != second_epoch

    def test_rank_offset_preserved_under_shuffle(self):
        # Rank 0 sees the same total set as rank 1, just a different
        # subset. Under shuffle, the offset is still rank % world_size.
        rows = self._rows(8)
        idx0 = ShardIndex(rows, rank=0, world_size=2, shuffle=True, seed=0)
        idx1 = ShardIndex(rows, rank=1, world_size=2, shuffle=True, seed=0)
        # Each rank sees half the shards
        assert sum(1 for _ in idx0) == 4
        assert sum(1 for _ in idx1) == 4
        # The un-shuffled order would be [0,2,4,6] for rank 0 and
        # [1,3,5,7] for rank 1. After shuffle we just need disjoint sets.
        assert set(s.path for s in idx0).isdisjoint(set(s.path for s in idx1))


# ── AsyncShardLoader (sync path) ──────────────────────────────────────────
class TestAsyncShardLoaderSync:
    def _make_manifest(self, tmp_path: Path, n_tokens: int = 1000) -> Path:
        tokens = torch.arange(n_tokens, dtype=torch.long)
        manifest = write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=400,
            max_seq_len=10,
            pad_token_id=0,
        )
        return write_manifest(manifest, tmp_path)

    def test_iter_yields_correct_shape(self, tmp_path: Path):
        path = self._make_manifest(tmp_path, n_tokens=1000)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=2,
            grad_accum=1,
            seqlen=20,
            async_mode=False,
            device="cpu",
        )
        it = iter(loader)
        tokens, targets = next(it)
        assert tokens.shape == (2, 20)
        assert targets.shape == (2, 20)
        assert tokens.dtype == torch.long
        assert targets.dtype == torch.long

    def test_targets_are_left_shift(self, tmp_path: Path):
        path = self._make_manifest(tmp_path, n_tokens=200)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=1,
            grad_accum=1,
            seqlen=10,
            async_mode=False,
            device="cpu",
        )
        tokens, targets = next(iter(loader))
        # Last column of targets is -100
        assert (targets[:, -1] == -100).all()
        # Other columns match left-shift
        assert torch.equal(targets[:, :-1], tokens[:, 1:])

    def test_iteration_is_finite_per_shard(self, tmp_path: Path):
        # 1000 tokens at seqlen=20, batch=2 → 50 micro-batches possible
        path = self._make_manifest(tmp_path, n_tokens=1000)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=2,
            grad_accum=1,
            seqlen=20,
            async_mode=False,
            device="cpu",
        )
        count = 0
        for _ in loader:
            count += 1
            if count > 100:
                break
        # At least one micro-batch per shard
        assert count >= 3

    def test_set_batch_size_works(self, tmp_path: Path):
        path = self._make_manifest(tmp_path, n_tokens=400)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=2,
            grad_accum=1,
            seqlen=20,
            async_mode=False,
            device="cpu",
        )
        loader.set_batch_size(4)
        assert loader.batch_size == 4
        tokens, _ = next(iter(loader))
        assert tokens.shape[0] == 4  # batch dim is 4 now

    def test_set_seq_len_works(self, tmp_path: Path):
        path = self._make_manifest(tmp_path, n_tokens=400)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=2,
            grad_accum=1,
            seqlen=20,
            async_mode=False,
            device="cpu",
        )
        loader.set_seq_len(40)
        assert loader.seqlen == 40
        tokens, _ = next(iter(loader))
        assert tokens.shape[1] == 40

    def test_stats_shape(self, tmp_path: Path):
        path = self._make_manifest(tmp_path, n_tokens=400)
        loader = AsyncShardLoader(
            path,
            root=tmp_path,
            batch_size=2,
            grad_accum=1,
            seqlen=20,
            async_mode=False,
            device="cpu",
        )
        s = loader.stats()
        assert s["rank"] == 0
        assert s["world_size"] == 1
        assert s["n_shards"] == 1
        assert s["total_tokens"] == 400
        assert s["async_mode"] is False
        assert s["device"] == "cpu"

    def test_empty_manifest_raises(self, tmp_path: Path):
        # Write an empty manifest
        empty = tmp_path / "shards" / "manifest.jsonl"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_text("")
        with pytest.raises(ValueError, match="No shards"):
            AsyncShardLoader(
                empty,
                root=tmp_path,
                batch_size=2,
                grad_accum=1,
                seqlen=20,
                async_mode=False,
                device="cpu",
            )
