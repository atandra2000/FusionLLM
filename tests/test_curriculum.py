"""Unit tests for `data/curriculum.py`.

Phase 1.6 scope:
  * STAGE_1_WEIGHTS / STAGE_2_WEIGHTS sum to 1.0
  * CurriculumStage: alias sampler, exclusion of zero-weight
    sources, in_scope_sources
  * Curriculum: stage 1 default, advance() at switch_step,
    stats(), sampling returns shards only from in-scope sources
"""

from __future__ import annotations

import pytest

from data.async_loader import ShardMeta
from data.curriculum import (
    STAGE_1_WEIGHTS,
    STAGE_2_WEIGHTS,
    Curriculum,
    CurriculumStage,
)


# ── Canonical mix ──────────────────────────────────────────────────────────
class TestStageWeights:
    def test_stage_1_sums_to_one(self):
        assert abs(sum(STAGE_1_WEIGHTS.values()) - 1.0) < 1e-9

    def test_stage_2_sums_to_one(self):
        assert abs(sum(STAGE_2_WEIGHTS.values()) - 1.0) < 1e-9

    def test_stage_2_has_more_code_and_math(self):
        # Stage 2 should be more code/math-heavy than stage 1
        s1_code_math = STAGE_1_WEIGHTS.get("stack_edu", 0) + STAGE_1_WEIGHTS.get("openr1_math", 0)
        s2_code_math = STAGE_2_WEIGHTS.get("stack_edu", 0) + STAGE_2_WEIGHTS.get("openr1_math", 0)
        assert s2_code_math > s1_code_math


# ── CurriculumStage ────────────────────────────────────────────────────────
class TestCurriculumStage:
    def _shards(self) -> list:
        return [
            ShardMeta(path="s0.bin", n_tokens=100, source="fineweb_edu"),
            ShardMeta(path="s1.bin", n_tokens=100, source="stack_edu"),
            ShardMeta(path="s2.bin", n_tokens=100, source="openr1_math"),
            ShardMeta(path="s3.bin", n_tokens=100, source="fineweb2"),
        ]

    def test_in_scope_sources(self):
        stage = CurriculumStage(
            name="x",
            weights=STAGE_1_WEIGHTS,
            shards=self._shards(),
        )
        # Stage 1 has no smollm_corpus
        assert "smollm_corpus" not in stage.in_scope_sources
        assert "fineweb_edu" in stage.in_scope_sources

    def test_empty_shards(self):
        stage = CurriculumStage(name="x", weights=STAGE_1_WEIGHTS, shards=[])
        assert len(stage) == 0
        with pytest.raises(RuntimeError, match="no shards"):
            import random

            stage.sample(random.Random(0))

    def test_zero_total_weight_raises(self):
        # Stage with no shards in any of its in-scope sources
        shards = [ShardMeta(path="s.bin", n_tokens=100, source="smollm_corpus")]
        with pytest.raises(RuntimeError, match="zero total weight"):
            CurriculumStage(
                name="x",
                weights=STAGE_1_WEIGHTS,
                shards=shards,
            ).sample(__import__("random").Random(0))

    def test_sample_returns_in_scope_source(self):
        import random

        stage = CurriculumStage(
            name="x",
            weights=STAGE_1_WEIGHTS,
            shards=self._shards(),
        )
        # Sample many times; none should be smollm_corpus
        rng = random.Random(0)
        for _ in range(100):
            s = stage.sample(rng)
            assert s.source in stage.in_scope_sources

    def test_sample_deterministic_with_seed(self):
        import random

        stage = CurriculumStage(
            name="x",
            weights=STAGE_1_WEIGHTS,
            shards=self._shards(),
        )
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        for _ in range(20):
            assert stage.sample(rng1).path == stage.sample(rng2).path


# ── Curriculum (top-level) ────────────────────────────────────────────────
class TestCurriculum:
    def _shards(self) -> list:
        return [
            ShardMeta(path="web0.bin", n_tokens=1000, source="fineweb_edu"),
            ShardMeta(path="web1.bin", n_tokens=1000, source="fineweb_edu"),
            ShardMeta(path="code0.bin", n_tokens=1000, source="stack_edu"),
            ShardMeta(path="math0.bin", n_tokens=1000, source="openr1_math"),
            ShardMeta(path="ml0.bin", n_tokens=1000, source="smollm_corpus"),
        ]

    def _make_manifest(self, tmp_path) -> object:
        import json
        from pathlib import Path

        p = tmp_path / "manifest.jsonl"
        rows = []
        for s in self._shards():
            rows.append(
                {
                    "path": s.path,
                    "n_tokens": s.n_tokens,
                    "source": s.source,
                    "weight": 1.0,
                    "quality_score": 1.0,
                    "domain": s.source,
                }
            )
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def test_active_is_stage_1_by_default(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=100)
        assert c.active.name == "stage_1"

    def test_advance_at_switch_step(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=100, seed=0)

        # Before switch: no change
        assert c.advance(50) is False
        assert c.active.name == "stage_1"

        # At switch: change
        assert c.advance(100) is True
        assert c.active.name == "stage_2"

        # After switch: no further change
        assert c.advance(200) is False
        assert c.active.name == "stage_2"

    def test_advance_before_switch(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=100, seed=0)
        assert c.advance(99) is False
        assert c.active.name == "stage_1"

    def test_sample_uses_active_stage(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=0, seed=0)
        c.advance(0)  # trigger the hot-swap to stage 2
        # Stage 2 includes smollm_corpus
        sources = set()
        for _ in range(200):
            s = c.sample()
            sources.add(s.source)
        assert "smollm_corpus" in sources

    def test_stats_shape(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=100, seed=0)
        s = c.stats()
        assert s["active"] == "stage_1"
        assert s["switch_step"] == 100
        assert s["advanced"] is False
        assert "fineweb_edu" in s["stage_1_in_scope"]
        assert "smollm_corpus" not in s["stage_1_in_scope"]
        assert "smollm_corpus" in s["stage_2_in_scope"]

    def test_iter_active_filters_by_scope(self, tmp_path):
        path = self._make_manifest(tmp_path)
        c = Curriculum(path, switch_step=0, seed=0)
        c.advance(0)  # trigger the hot-swap to stage 2
        # Stage 2 is active; smollm_corpus should be in-scope
        active = c.iter_active()
        sources = [s.source for s in active]
        assert "smollm_corpus" in sources

    def test_empty_manifest_raises(self, tmp_path):
        from pathlib import Path

        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="No shards"):
            Curriculum(p, switch_step=100)
