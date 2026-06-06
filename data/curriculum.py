"""Curriculum manifest and 2-stage sampler.

The pre-training corpus is split into two stages (modded-nanogpt
"anneal on high-skill data" trick from OLMo-2 / Qwen2.5):

* **Stage 1 (0 → 70 %)** — web-heavy:
  ``{"fineweb_edu": 0.70, "stack_edu": 0.15, "openr1_math": 0.05,
     "fineweb2": 0.10}``

* **Stage 2 (70 → 100 %)** — code/math-heavy anneal:
  ``{"fineweb_edu": 0.30, "stack_edu": 0.25, "openr1_math": 0.25,
     "fineweb2": 0.10, "smollm_corpus": 0.10}``

Per-shard manifest
------------------
Each shard produced by ``data/prepare_data.write_shards`` has a row
in ``shards/manifest.jsonl`` with a ``source`` field.  The
:class:`Curriculum` reads that manifest and builds per-stage
weighted samplers.

Public surface
--------------
* :class:`Curriculum` — top-level entry point.
* :class:`CurriculumStage` — one stage of the curriculum.
* :func:`STAGE_1_WEIGHTS` / :func:`STAGE_2_WEIGHTS` — the canonical
  SmolTalk-3-inspired mix.
* :func:`advance(step)` — hot-swap the active stage at a step
  boundary.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from data.async_loader import ShardMeta, load_manifest

# ── Canonical mix (SmolTalk-3 inspired) ───────────────────────────────────
# The two stages sum to 1.0 each.  Sources not in a stage are
# excluded at runtime (a shard with an excluded source is skipped
# by the sampler).
STAGE_1_WEIGHTS: dict[str, float] = {
    "fineweb_edu": 0.70,
    "stack_edu": 0.15,
    "openr1_math": 0.05,
    "fineweb2": 0.10,
}

STAGE_2_WEIGHTS: dict[str, float] = {
    "fineweb_edu": 0.30,
    "stack_edu": 0.25,
    "openr1_math": 0.25,
    "fineweb2": 0.10,
    "smollm_corpus": 0.10,
}


# ── Per-stage sampler ──────────────────────────────────────────────────────
@dataclass
class CurriculumStage:
    """One stage of the curriculum.

    Holds a list of ``ShardMeta`` (all of them) plus a sampling
    weight per shard.  Sampling is **with replacement** proportional
    to weight, which is the resampling-aware packing trick from
    modded-nanogpt #33.
    """

    name: str
    weights: dict[str, float]
    shards: list[ShardMeta] = field(default_factory=list)
    _shard_weights: list[float] = field(default_factory=list)
    _alias: tuple[list[int], list[float]] | None = None
    _alias_n: int = 0

    def __post_init__(self) -> None:
        if not self.shards:
            return
        # Per-shard weight = stage weight of the source × shard quality.
        # (Shard quality lives in ``ShardMeta.quality_score``; default
        # is 1.0 in the writer.)
        self._shard_weights = []
        for s in self.shards:
            if s.source == "mixed":
                # Mixed-source shards: use uniform weight when no
                # per-source metadata is available.
                w = sum(self.weights.values()) / max(len(self.weights), 1)
            else:
                w = self.weights.get(s.source, 0.0)
            if w <= 0.0:
                self._shard_weights.append(0.0)
            else:
                self._shard_weights.append(w * s.quality_score)
        # Pre-build an alias table for O(1) sampling (Vose alias
        # method).  We rebuild on the first sample so the table
        # only exists when actually used.
        self._alias = None

    def __len__(self) -> int:
        return len(self.shards)

    def sample(self, rng: random.Random) -> ShardMeta:
        """Sample one shard with replacement, weighted by stage weight."""
        if not self.shards:
            raise RuntimeError("CurriculumStage has no shards")
        if self._alias is None:
            self._build_alias()
        # Vose alias: pick bucket, then decide
        i = rng.randrange(self._alias_n)
        if rng.random() < self._alias[1][i]:
            return self.shards[self._alias[0][i]]
        return self.shards[i]

    def _build_alias(self) -> None:
        """Build the Vose alias table over the per-shard weights."""
        n = len(self._shard_weights)
        total = sum(self._shard_weights)
        if total <= 0:
            raise RuntimeError(f"Stage {self.name!r} has zero total weight — no shard is in-scope")
        # Normalise
        probs = [w / total for w in self._shard_weights]
        # Standard Vose: O(n) construction.
        small: list[int] = []
        large: list[int] = []
        for i, p in enumerate(probs):
            (small if p < 1.0 / n else large).append(i)
        alias = [0] * n
        prob = [0.0] * n
        while small and large:
            s = small.pop()
            l = large.pop()
            prob[s] = probs[s] * n
            alias[s] = l
            probs[l] = probs[l] + probs[s] - 1.0 / n
            if probs[l] < 1.0 / n:
                small.append(l)
            else:
                large.append(l)
        while large:
            l = large.pop()
            prob[l] = 1.0
        while small:
            s = small.pop()
            prob[s] = 1.0
        self._alias = (alias, prob)
        self._alias_n = n

    @property
    def in_scope_sources(self) -> list[str]:
        return [s for s, w in self.weights.items() if w > 0]


# ── Top-level curriculum ──────────────────────────────────────────────────
class Curriculum:
    """Two-stage curriculum over the sharded corpus.

    Args:
        manifest_path: path to ``shards/manifest.jsonl``.
        stage_1_weights: stage 1 mix (default :data:`STAGE_1_WEIGHTS`).
        stage_2_weights: stage 2 mix (default :data:`STAGE_2_WEIGHTS`).
        switch_step: when to advance from stage 1 to stage 2.
        seed: PRNG seed.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        stage_1_weights: dict[str, float] | None = None,
        stage_2_weights: dict[str, float] | None = None,
        switch_step: int = 0,
        seed: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        all_shards = load_manifest(self.manifest_path)
        if not all_shards:
            raise ValueError(f"No shards in {self.manifest_path}")

        self.stage_1 = CurriculumStage(
            name="stage_1",
            weights=dict(stage_1_weights or STAGE_1_WEIGHTS),
            shards=all_shards,
        )
        self.stage_2 = CurriculumStage(
            name="stage_2",
            weights=dict(stage_2_weights or STAGE_2_WEIGHTS),
            shards=all_shards,
        )
        self.switch_step = int(switch_step)
        self._rng = random.Random(seed)
        self._active: CurriculumStage = self.stage_1
        self._advanced: bool = False

    @property
    def active(self) -> CurriculumStage:
        return self._active

    def advance(self, step: int) -> bool:
        """Hot-swap to stage 2 if ``step >= switch_step`` and not already done.

        Returns ``True`` if the active stage changed this call.
        """
        if not self._advanced and step >= self.switch_step:
            self._active = self.stage_2
            self._advanced = True
            return True
        return False

    def sample(self) -> ShardMeta:
        """Sample one shard from the active stage."""
        return self._active.sample(self._rng)

    def iter_active(self) -> list[ShardMeta]:
        """Return a *list* of shards in the active stage's in-scope sources.

        Used by the trainer to compute a per-epoch ordering.
        """
        return [s for s in self._active.shards if s.source in self._active.in_scope_sources]

    def stats(self) -> dict:
        return {
            "active": self._active.name,
            "switch_step": self.switch_step,
            "advanced": self._advanced,
            "stage_1_in_scope": self.stage_1.in_scope_sources,
            "stage_2_in_scope": self.stage_2.in_scope_sources,
            "n_shards": len(self.stage_1.shards),
        }
