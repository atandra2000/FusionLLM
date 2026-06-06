"""Unit tests for `data/dedup.py`.

Phase 1.1 scope:
  * exact_prefix_dedup
  * md5_fallback
  * MinHashHasher — signature shape, determinism, empty doc
  * LSHIndex — add + candidates round-trip
  * deduplicate_docs — top-level entry point (3 strategies, cluster
    selection, threshold, empty input)
"""

from __future__ import annotations

import numpy as np
import pytest

from data.dedup import (
    LSHIndex,
    MinHashHasher,
    deduplicate_docs,
    exact_prefix_dedup,
    md5_fallback,
    md5_fallback_key,
)


# ── exact_prefix_dedup ─────────────────────────────────────────────────────
class TestExactPrefixDedup:
    def test_identical_prefix_dropped(self):
        # Use a 20-char prefix that is *byte-for-byte* identical.
        docs = [
            ("0123456789abcdefghij" + "x" * 200, 1.0),
            ("0123456789abcdefghij" + "y" * 200, 0.5),
        ]
        out = exact_prefix_dedup(docs, n=20)
        assert len(out) == 1
        # Longer one wins
        assert out[0][0].endswith("x" * 200)

    def test_different_prefixes_kept(self):
        docs = [
            ("alpha " * 50, 1.0),
            ("beta  " * 50, 1.0),
        ]
        out = exact_prefix_dedup(docs, n=20)
        assert len(out) == 2

    def test_zero_bytes_is_passthrough(self):
        docs = [("a", 1.0), ("a", 1.0)]
        assert exact_prefix_dedup(docs, n=0) == docs

    def test_empty_list(self):
        assert exact_prefix_dedup([]) == []


# ── md5_fallback ───────────────────────────────────────────────────────────
class TestMd5Fallback:
    def test_same_first_words_dropped(self):
        # 200 unique words, both with the same first 128.
        common = " ".join(f"w{i}" for i in range(128))
        docs = [
            (common + " " + " ".join(f"x{i}" for i in range(200)), 1.0),
            (common + " " + " ".join(f"y{i}" for i in range(200)), 1.0),
        ]
        out = md5_fallback(docs)
        assert len(out) == 1
        assert out[0][0].startswith(common)

    def test_different_first_words_kept(self):
        docs = [
            ("first sentence " * 50, 1.0),
            ("completely different text " * 50, 1.0),
        ]
        out = md5_fallback(docs)
        assert len(out) == 2

    def test_key_is_deterministic(self):
        a = md5_fallback_key("the quick brown fox")
        b = md5_fallback_key("the quick brown fox")
        assert a == b

    def test_key_length_32(self):
        # MD5 hex
        assert len(md5_fallback_key("hello world")) == 32


# ── MinHashHasher ──────────────────────────────────────────────────────────
class TestMinHashHasher:
    def test_signature_shape(self):
        h = MinHashHasher(num_perm=1000, ngram=5)
        sig = h.signature("the quick brown fox jumps over the lazy dog " * 5)
        assert sig.shape == (1000,)
        assert sig.dtype == np.int64

    def test_signature_deterministic(self):
        h1 = MinHashHasher(num_perm=500, ngram=5, seed=42)
        h2 = MinHashHasher(num_perm=500, ngram=5, seed=42)
        text = "the quick brown fox " * 20
        assert np.array_equal(h1.signature(text), h2.signature(text))

    def test_signature_different_seeds(self):
        h1 = MinHashHasher(num_perm=500, ngram=5, seed=0)
        h2 = MinHashHasher(num_perm=500, ngram=5, seed=1)
        text = "the quick brown fox " * 20
        # Different seeds → different permutations → different signatures
        assert not np.array_equal(h1.signature(text), h2.signature(text))

    def test_empty_doc_returns_max(self):
        h = MinHashHasher(num_perm=200, ngram=5)
        sig = h.signature("")
        assert sig.shape == (200,)
        # All entries are INT64_MAX (no shingles to take min over)
        assert (sig == np.iinfo(np.int64).max).all()

    def test_short_doc_returns_one_shingle(self):
        h = MinHashHasher(num_perm=200, ngram=5)
        sig = h.signature("hi there")  # 2 words, fewer than n=5
        # _shingles falls back to a single shingle of all available
        # words; signature is a valid min over one hash.
        assert sig.shape == (200,)
        assert (sig != np.iinfo(np.int64).max).any()

    def test_similar_docs_have_high_jaccard(self):
        h = MinHashHasher(num_perm=5000, ngram=5, seed=0)
        # Two long docs with >90% overlap of the *shingle set*.
        a_text = " ".join(f"w{i}" for i in range(200))  # 200 unique words
        b_text = a_text + " " + " ".join(f"extra{i}" for i in range(5))
        a = h.signature(a_text)
        b = h.signature(b_text)
        jaccard = (a == b).mean()
        # 5-gram shingles from 200 words: 196 shingles. Adding 5
        # unique words to the end adds only a few new shingles, so
        # Jaccard should be > 0.9.
        assert jaccard > 0.9, f"Expected high Jaccard, got {jaccard}"

    def test_dissimilar_docs_have_low_jaccard(self):
        h = MinHashHasher(num_perm=5000, ngram=5, seed=0)
        a = h.signature(" ".join(f"alpha{i}" for i in range(200)))
        b = h.signature(" ".join(f"beta{i}" for i in range(200)))
        jaccard = (a == b).mean()
        # Two disjoint vocabularies → near-zero Jaccard
        assert jaccard < 0.1


# ── LSHIndex ───────────────────────────────────────────────────────────────
class TestLSHIndex:
    def test_add_and_candidates_roundtrip(self):
        h = MinHashHasher(num_perm=1000, ngram=3, seed=0)
        idx = LSHIndex(num_perm=1000, num_bands=20)
        sig = h.signature("a b c d e f g h " * 20)
        idx.add(0, sig)
        # Same signature → must be a candidate
        cand = list(idx.candidates(1, sig))
        assert 0 in cand

    def test_candidates_skip_self(self):
        h = MinHashHasher(num_perm=1000, ngram=3, seed=0)
        idx = LSHIndex(num_perm=1000, num_bands=20)
        sig = h.signature("a b c d e f g h " * 20)
        idx.add(7, sig)
        # Same doc-id should not be returned
        cand = list(idx.candidates(7, sig))
        assert 7 not in cand

    def test_num_perm_must_divide_bands(self):
        with pytest.raises(ValueError, match="must be divisible"):
            LSHIndex(num_perm=1000, num_bands=63)


# ── deduplicate_docs (top-level) ──────────────────────────────────────────
class TestDeduplicateDocs:
    def test_empty(self):
        assert deduplicate_docs([]) == []
        assert deduplicate_docs([], strategy="prefix") == []
        assert deduplicate_docs([], strategy="md5") == []

    def test_single_doc_passthrough(self):
        docs = [("only one document", 1.0)]
        for strat in ("minhash", "prefix", "md5"):
            assert deduplicate_docs(docs, strategy=strat) == docs

    def test_exact_duplicates_collapsed(self):
        docs = [
            ("the same text repeated for many words " * 30, 1.0),
            ("the same text repeated for many words " * 30, 0.5),
        ]
        out = deduplicate_docs(docs, num_perm=2000, num_bands=20, min_dup_chars=10)
        assert len(out) == 1
        # First-seen wins (tie-break on length also picks first)
        assert out[0][1] == 1.0

    def test_near_duplicates_collapsed_longest_kept(self):
        # Two long docs that differ only in a few words — high
        # shingle Jaccard.
        common = " ".join(f"w{i}" for i in range(200))
        docs = [
            (common, 0.5),  # short
            (common + " w42 w43 w44 w99", 1.0),  # tiny edit
            (" ".join(f"unrelated{i}" for i in range(200)), 0.5),  # different
        ]
        out = deduplicate_docs(
            docs,
            num_perm=2000,
            num_bands=20,
            min_dup_chars=10,
        )
        assert len(out) == 2
        surviving = [t for t, _ in out]
        # The longer one (doc 1) should survive; doc 0 deduped
        assert any("w42 w43 w44 w99" in s for s in surviving)
        assert not any(s == docs[0][0] for s in surviving)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown dedup strategy"):
            deduplicate_docs([("hello world", 1.0)], strategy="does-not-exist")

    def test_short_docs_skipped_from_minhash(self):
        # Two docs both shorter than min_dup_chars → kept as-is
        # (MinHash path is skipped; the prefix dedup would still
        # catch them, but the minhash strategy does not).
        docs = [("a b c", 1.0), ("a b c", 1.0)]
        out = deduplicate_docs(
            docs,
            num_perm=2000,
            num_bands=20,
            min_dup_chars=100,
        )
        # MinHash skipped both; no dedup happens for short docs
        assert len(out) == 2

    def test_strategy_prefix(self):
        docs = [
            ("alpha " * 50, 1.0),
            ("alpha " * 50 + " tail", 0.5),
            ("beta " * 50, 1.0),
        ]
        out = deduplicate_docs(docs, strategy="prefix", prefix_bytes=20)
        assert len(out) == 2

    def test_strategy_md5(self):
        common = " ".join(f"w{i}" for i in range(128))
        docs = [
            (common + " " + " ".join(f"x{i}" for i in range(200)), 1.0),
            (common + " " + " ".join(f"y{i}" for i in range(200)), 1.0),
        ]
        out = deduplicate_docs(docs, strategy="md5")
        assert len(out) == 1
