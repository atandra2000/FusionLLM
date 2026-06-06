"""Tests for Phase 6.3 — curriculum hot-swap + loader reconfiguration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from data.async_loader import AsyncShardLoader, ShardMeta
from data.curriculum import Curriculum


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_shard(path: Path, n_int32: int) -> None:
    """Write a valid shard file with *n_int32* dummy token slots."""
    header = bytearray(256)
    header[0] = 0x01  # version = 1
    header[1:9] = n_int32.to_bytes(8, "little")  # n_tokens
    header[9:13] = (4096).to_bytes(4, "little")   # max_seq_len
    data = np.arange(n_int32, dtype=np.int32).tobytes()
    path.write_bytes(bytes(header) + data)


def _make_manifest(tmpdir: str, n_shards: int = 10) -> str:
    path = Path(tmpdir) / "shards" / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_shards):
            shard_path = Path(tmpdir) / "shards" / f"shard_{i:04d}.bin"
            _make_shard(shard_path, n_int32=2 * 4 * 6)  # 6 micro-batches worth
            row = {
                "path": str(shard_path),
                "n_tokens": 100_000,
                "source": "fineweb_edu" if i < 7 else "stack_edu",
                "weight": 1.0,
                "quality_score": 1.0,
            }
            f.write(json.dumps(row) + "\n")
    return str(path)


class TestShardIndexSetShards:
    def test_set_shards_replaces_shards(self):
        shards_a = [ShardMeta(path=f"a_{i}", n_tokens=100) for i in range(5)]
        shards_b = [ShardMeta(path=f"b_{i}", n_tokens=100) for i in range(3)]
        from data.async_loader import ShardIndex

        idx = ShardIndex(shards_a, rank=0, world_size=1)
        assert len(idx) == 5
        idx.set_shards(shards_b)
        assert len(idx) == 3
        assert all(s.path.startswith("b_") for s in idx.shards)


class TestAsyncShardLoaderSetShards:
    @pytest.fixture
    def manifest(self, tmp_path):
        return _make_manifest(str(tmp_path))

    def test_set_shards_updates_loader(self, manifest):
        loader = AsyncShardLoader(
            Path(manifest),
            batch_size=2,
            grad_accum=1,
            seqlen=4,
            async_mode=False,
        )
        original_count = len(loader.shards)
        subset = loader.shards[:3]
        loader.set_shards(subset)
        assert len(loader.shards) == 3
        assert len(loader.index) == 3

    def test_set_shards_preserves_iteration(self, manifest):
        loader = AsyncShardLoader(
            Path(manifest),
            batch_size=2,
            grad_accum=1,
            seqlen=4,
            async_mode=False,
            seed=42,
        )
        all_shards = list(loader.shards)
        subset = all_shards[:2]
        loader.set_shards(subset)
        tokens, targets = next(iter(loader))
        assert tokens.shape == (2, 4)
        assert targets.shape == (2, 4)


class TestCurriculumIntegration:
    def test_curriculum_advance_triggers_switch(self, tmp_path):
        from data.curriculum import STAGE_1_WEIGHTS, STAGE_2_WEIGHTS

        manifest = _make_manifest(str(tmp_path))
        curriculum = Curriculum(
            manifest_path=Path(manifest),
            stage_1_weights=STAGE_1_WEIGHTS,
            stage_2_weights=STAGE_2_WEIGHTS,
            switch_step=10,
            seed=0,
        )
        assert curriculum.active.name == "stage_1"
        assert not curriculum.advance(5)
        assert curriculum.active.name == "stage_1"
        assert curriculum.advance(10)
        assert curriculum.active.name == "stage_2"
        assert not curriculum.advance(15)  # already advanced

    def test_iter_active_returns_subset(self, tmp_path):
        manifest = _make_manifest(str(tmp_path))
        curriculum = Curriculum(
            manifest_path=Path(manifest),
            switch_step=10,
            seed=0,
        )
        active = curriculum.iter_active()
        assert len(active) > 0
        for s in active:
            assert s.source in curriculum.stage_1.in_scope_sources

    def test_curriculum_stats(self, tmp_path):
        manifest = _make_manifest(str(tmp_path))
        curriculum = Curriculum(manifest_path=Path(manifest), switch_step=10)
        stats = curriculum.stats()
        assert stats["active"] == "stage_1"
        assert stats["switch_step"] == 10
        assert not stats["advanced"]
        curriculum.advance(10)
        stats = curriculum.stats()
        assert stats["active"] == "stage_2"
        assert stats["advanced"]


class TestSchedulesReconfigureLoader:
    def test_batch_size_reconfigures_loader(self):
        from data.async_loader import AsyncShardLoader, ShardMeta

        shards = [ShardMeta(path=f"test_{i}", n_tokens=1000) for i in range(3)]
        loader = AsyncShardLoader.__new__(AsyncShardLoader)
        loader.shards = shards
        loader.batch_size = 2
        loader.seqlen = 4
        loader._mb_tokens = 8
        loader.async_mode = False

        loader.set_batch_size(4)
        assert loader.batch_size == 4
        assert loader._mb_tokens == 16

    def test_seq_len_reconfigures_loader(self):
        from data.async_loader import AsyncShardLoader, ShardMeta

        shards = [ShardMeta(path=f"test_{i}", n_tokens=1000) for i in range(3)]
        loader = AsyncShardLoader.__new__(AsyncShardLoader)
        loader.shards = shards
        loader.batch_size = 2
        loader.seqlen = 4
        loader._mb_tokens = 8
        loader.async_mode = False

        loader.set_seq_len(8)
        assert loader.seqlen == 8
        assert loader._mb_tokens == 16
