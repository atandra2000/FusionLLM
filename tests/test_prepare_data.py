"""Unit tests for `data/prepare_data.py`.

Phase 1.2 + 1.3 + 1.4 scope:
  * Configuration: MAX_DOCS, QUALITY_THRESHOLD, EOS_TEXT.
  * Text filters: normalize, passes_language_filter, quality_score.
  * Source dispatcher: ``_COLLECTORS`` keys + unknown-source error.
  * Curriculum mix: orders docs by descending quality.
  * Block-pack: tokenize_and_pack shape + packing efficiency
    (uses a tiny tokenizer-free path; the real tokenizer test
    lives in `tests/test_dedup.py`).
  * Sharded mmap writer: round-trip on a small tensor.

The HF `datasets.load_dataset` collectors are NOT tested here (they
require network + the `datasets` package, gated by the heavy-deps
extra in `pyproject.toml`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from data.prepare_data import (
    _COLLECTORS,
    DEFAULT_MAX_SEQ_LEN,
    EOS_TEXT,
    MAX_DOCS,
    MINHASH_NUM_BANDS,
    MINHASH_NUM_PERM,
    QUALITY_THRESHOLD,
    TARGET_SHARD_TOKENS,
    _filter_docs,
    curriculum_mix,
    deduplicate,
    export_eval_samples,
    normalize,
    passes_language_filter,
    quality_score,
    tokenize_and_pack,
    write_manifest,
    write_shards,
)


# ── Configuration ──────────────────────────────────────────────────────────
class TestConfig:
    def test_seven_sources(self):
        assert len(MAX_DOCS) == 7
        assert set(MAX_DOCS) == {
            "fineweb_edu",
            "finemath",
            "stack_edu",
            "cosmopedia",
            "openr1_math",
            "fineweb2",
            "smollm_corpus",
        }

    def test_collectors_match_max_docs(self):
        assert set(_COLLECTORS) == set(MAX_DOCS)

    def test_every_source_has_quality_threshold(self):
        for src in MAX_DOCS:
            assert src in QUALITY_THRESHOLD, f"missing threshold for {src}"

    def test_minhash_defaults(self):
        assert MINHASH_NUM_PERM == 1_000_000
        assert MINHASH_NUM_BANDS == 64

    def test_default_max_seq_len_is_8k(self):
        assert DEFAULT_MAX_SEQ_LEN == 8192

    def test_target_shard_is_4_b_tokens(self):
        assert TARGET_SHARD_TOKENS == 4_000_000_000


# ── Text filters ───────────────────────────────────────────────────────────
class TestNormalize:
    def test_collapses_whitespace(self):
        assert normalize("a  b\n\nc\td") == "a b c d"

    def test_strips(self):
        assert normalize("  hello  ") == "hello"


class TestLanguageFilter:
    def test_pure_english_passes(self):
        assert passes_language_filter("the quick brown fox")

    def test_pure_chinese_fails(self):
        # 85 % non-ASCII check; Chinese characters are not ASCII.
        text = "中文测试文本" * 20
        assert not passes_language_filter(text)

    def test_empty_text_fails(self):
        assert not passes_language_filter("")

    def test_mostly_english_passes(self):
        # 90 % ASCII characters
        text = ("hello " * 18) + "x"  # >85% ASCII
        assert passes_language_filter(text)


class TestQualityScore:
    def test_zero_for_too_short(self):
        assert quality_score("hi") == 0.0
        assert quality_score("a" * 119) == 0.0  # < 120 chars

    def test_zero_for_too_few_words(self):
        # ≥ 120 chars but < 20 words
        text = "x" * 130
        assert quality_score(text) == 0.0

    def test_length_score_caps_at_one(self):
        # 200+ words → length_score == 1.0
        text = " ".join(f"w{i}" for i in range(300))
        score = quality_score(text)
        assert 1.0 < score <= 2.0

    def test_low_diversity_lowers_score(self):
        # 100 words but only 5 unique (5 % diversity) → score
        # = length_score(100/200) + diversity_score(5/100) = 0.55
        text = " ".join(["a", "b", "c", "d", "e"] * 20)
        assert len(text.split()) == 100
        score = quality_score(text)
        assert score < 0.6


# ── Source dispatcher ──────────────────────────────────────────────────────
class TestSourceDispatcher:
    def test_unknown_source_raises(self):
        from data.prepare_data import collect

        with pytest.raises(ValueError, match="Unknown source"):
            collect("does-not-exist", 10)

    def test_known_sources_in_registry(self):
        for src in MAX_DOCS:
            assert src in _COLLECTORS
            assert callable(_COLLECTORS[src])


# ── Curriculum mix ─────────────────────────────────────────────────────────
class TestCurriculumMix:
    def test_concatenates_per_source(self):
        by_source = {
            "fineweb_edu": [("a", 0.5), ("b", 1.5)],
            "openr1_math": [("c", 1.0)],
        }
        out = curriculum_mix(by_source)
        # Per-source sort by descending quality: [b, a] + [c]
        assert [t for t, _ in out] == ["b", "a", "c"]

    def test_empty_input(self):
        assert curriculum_mix({}) == []


# ── Block-pack (tokenize_and_pack) ─────────────────────────────────────────
class TestBlockPack:
    """Tokenize-free tests using a fake tokenizer."""

    def _fake_tok(self):
        class Tok:
            pad_token_id = 0
            eos_token_id = 9

            def encode(self, text, add_special_tokens=True):
                # Map each word to a unique int 1..vocab
                return [hash(w) % 1000 + 1 for w in text.split()]

        return Tok()

    def test_basic_shape(self):
        tok = self._fake_tok()
        docs = [("a b c d e f g h " * 50, 1.0)] * 4  # 4 long docs
        tokens, eff = tokenize_and_pack(docs, tok, max_seq_len=64, min_efficiency=0, desc="x")
        assert tokens.dtype == torch.long
        assert tokens.numel() % 64 == 0
        assert 0.5 < eff <= 1.0

    def test_packing_efficiency_high(self):
        tok = self._fake_tok()
        # 10 docs each of ~30 tokens → fills 5 packs of 64 with high util
        docs = [(" ".join(f"w{i}" for i in range(30)), 1.0) for _ in range(10)]
        tokens, eff = tokenize_and_pack(docs, tok, max_seq_len=64, min_efficiency=0, desc="x")
        # Total tokens = 10*30 = 300, packed into 64-token sequences
        # → ceil(300/64) = 5 sequences → efficiency 300/(5*64) = 0.9375
        assert eff > 0.9

    def test_pads_final_buffer(self):
        tok = self._fake_tok()
        # 5 words → 5 tokens + 1 EOS = 6 tokens at max_seq_len=10
        # → 6/10 = 0.6 efficiency, padded to 10
        docs = [("a b c d e", 1.0)]
        tokens, eff = tokenize_and_pack(docs, tok, max_seq_len=10, min_efficiency=0, desc="x")
        assert tokens.numel() == 10
        assert eff == pytest.approx(0.6)

    def test_truncates_overlong_doc(self):
        tok = self._fake_tok()
        # One doc with 200 tokens; pack at 50 → truncated to 50
        docs = [(" ".join(f"w{i}" for i in range(200)), 1.0)]
        tokens, eff = tokenize_and_pack(docs, tok, max_seq_len=50, desc="x")
        assert tokens.numel() == 50


# ── Sharded mmap writer ────────────────────────────────────────────────────
class TestShardedWriter:
    def test_single_shard_round_trip(self, tmp_path: Path):
        # 100 tokens, target 1000 → 1 shard
        tokens = torch.arange(100, dtype=torch.long)
        manifest = write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=1000,
            max_seq_len=10,
            pad_token_id=-1,
        )
        assert len(manifest) == 1
        shard_path = tmp_path / "shards" / "shard_00000.bin"
        assert shard_path.exists()
        # Read back
        with open(shard_path, "rb") as f:
            header = f.read(256)
            data = f.read()
        assert header[0] == 0x01
        assert int.from_bytes(header[1:9], "little") == 100
        assert int.from_bytes(header[9:13], "little") == 10
        assert int.from_bytes(header[13:17], "little", signed=True) == -1
        # Data: 100 int32 values (400 bytes)
        assert len(data) == 100 * 4

    def test_multiple_shards(self, tmp_path: Path):
        # 250 tokens, target 100 → 3 shards (100, 100, 50)
        tokens = torch.arange(250, dtype=torch.long)
        manifest = write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=100,
            max_seq_len=10,
            pad_token_id=-1,
        )
        assert len(manifest) == 3
        assert [m["n_tokens"] for m in manifest] == [100, 100, 50]

    def test_manifest_written(self, tmp_path: Path):
        tokens = torch.arange(50, dtype=torch.long)
        manifest = write_shards(
            tokens,
            tmp_path,
            target_shard_tokens=100,
            max_seq_len=10,
            pad_token_id=0,
        )
        path = write_manifest(manifest, tmp_path)
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["n_tokens"] == 50
        assert rows[0]["path"] == "shards/shard_00000.bin"


# ── Eval samples ───────────────────────────────────────────────────────────
class TestEvalSamples:
    def test_writes_n_docs(self, tmp_path: Path):
        docs = [(f"document number {i}", float(i)) for i in range(10)]
        export_eval_samples(docs, tmp_path, n=3)
        out = (tmp_path / "eval_samples.txt").read_text()
        # 3 docs separated by 60-`=` lines
        assert out.count("=" * 60) == 3
        assert "document number 0" in out
        assert "document number 2" in out
        assert "document number 3" not in out


# ── Deduplicate wrapper ────────────────────────────────────────────────────
class TestDeduplicateWrapper:
    def test_dedup_strategy_kwarg_propagated(self):
        docs = [("hello world " * 10, 1.0), ("hello world " * 10, 1.0)]
        out = deduplicate(docs, strategy="prefix")
        # prefix dedup → both share the same first bytes → 1 doc
        assert len(out) == 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            deduplicate([("x", 1.0)], strategy="nope")
