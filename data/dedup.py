"""Deduplication primitives for the pre-training corpus.

Three strategies, in increasing cost and precision:

* :func:`exact_prefix_dedup` — match on the first ``n`` bytes of
  the document. Cheap, catches the obvious near-dups. O(1) per doc.
* :func:`md5_fallback` — match on the first 128 words, lowercased.
  Faster than MinHash for very small corpora; replaces the
  pre-Phase-1 ``near_duplicate_key`` helper in
  ``data/prepare_data.py``. O(1) per doc.
* :class:`MinHashHasher` + :class:`LSHIndex` — the real workhorse.
  1 M permutations, 5-gram shingle, 64 LSH bands.  Catches
  near-duplicates that the prefix check misses.  O(num_perm) per
  doc, O(num_bands) per candidate in the LSH.

Selection rule
--------------
When multiple duplicates are detected, the *longest* document is
kept (deepest content → most useful training signal).  Ties broken
by first-seen order.

Public surface
--------------
* :func:`deduplicate_docs(docs, *, strategy="minhash", num_perm=1_000_000,
  ngram=5, num_bands=64, prefix_bytes=128, min_dup_chars=120)`
* :class:`MinHashHasher`
* :class:`LSHIndex`
* :func:`exact_prefix_dedup`
* :func:`md5_fallback`
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import List, Optional, Tuple

import numpy as np

# ── Type aliases ────────────────────────────────────────────────────────────
Doc = tuple[str, float]  # (text, quality_score)


# ── Helpers ────────────────────────────────────────────────────────────────
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _shingles(text: str, n: int = 5) -> list[str]:
    """Tokenize on whitespace and yield ``n``-gram shingles as strings."""
    words = _normalize(text).lower().split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


# ── 1. Exact-prefix dedup ──────────────────────────────────────────────────
def exact_prefix_dedup(
    docs: Sequence[Doc],
    *,
    n: int = 128,
) -> list[Doc]:
    """Drop docs whose first ``n`` bytes match a previously seen doc.

    Cheap (O(1) hash per doc), catches scraper duplicates and
    identical reposts.  Two documents with the same prefix but
    different tails are treated as different.
    """
    if n <= 0:
        return list(docs)
    seen: set = set()
    out: list[Doc] = []
    for text, score in docs:
        key = text[:n].encode("utf-8", errors="ignore")
        h = hashlib.blake2b(key, digest_size=16).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append((text, score))
    return out


# ── 2. MD5-of-first-128-words fallback ─────────────────────────────────────
def md5_fallback_key(text: str, *, n_words: int = 128) -> str:
    """Match the pre-Phase-1 ``near_duplicate_key`` helper exactly.

    Kept for the ultra-fast smoke-test path; not used by the
    canonical pipeline.  Hashes the lowercased first ``n_words``
    words with MD5.
    """
    words = _normalize(text).lower().split()
    sample = " ".join(words[:n_words])
    return hashlib.md5(sample.encode()).hexdigest()


def md5_fallback(docs: Sequence[Doc], *, n_words: int = 128) -> list[Doc]:
    """MD5-of-first-128-words dedup.  Cheaper than MinHash, less precise."""
    seen: set = set()
    out: list[Doc] = []
    for text, score in docs:
        key = md5_fallback_key(text, n_words=n_words)
        if key in seen:
            continue
        seen.add(key)
        out.append((text, score))
    return out


# ── 3. MinHash + LSH (the canonical path) ─────────────────────────────────
class MinHashHasher:
    """Compute a MinHash signature for a document.

    The signature is a length-``num_perm`` int64 array; each entry
    is the minimum hash value of any shingle in the document, under
    that permutation.  Two documents are candidate near-duplicates
    when their signatures agree on enough positions (which is what
    :class:`LSHIndex` detects).

    Args:
        num_perm: number of permutations (default 1 000 000).
        ngram: shingle size in words (default 5).
        seed: PRNG seed (default 0 — must be stable for resume).
    """

    def __init__(self, num_perm: int = 1_000_000, ngram: int = 5, seed: int = 0):
        self.num_perm = num_perm
        self.ngram = ngram
        # Single 64-bit hash family: splitmix64.  We pre-compute the
        # ``a`` and ``b`` coefficients for the ``num_perm`` permutations
        # on first use.  The pair (a, b) is derived from the seed
        # so the signatures are stable.
        self._a: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._init_perms(seed)

    def _init_perms(self, seed: int) -> None:
        # 64-bit Mersenne-ish prime.  Using a smaller modulus here
        # would saturate the int64 range; 2**61-1 is standard.
        MOD = (1 << 61) - 1
        rng = np.random.default_rng(seed)
        # Coefficients in [1, MOD).
        self._a = rng.integers(1, MOD, size=self.num_perm, dtype=np.uint64)
        self._b = rng.integers(0, MOD, size=self.num_perm, dtype=np.uint64)
        self._mod = MOD

    def signature(self, text: str) -> np.ndarray:
        """Compute the MinHash signature for one document.

        Returns a length-``num_perm`` int64 array.  Empty documents
        return an all-``INT64_MAX`` signature (matches no other).
        """
        shingles = _shingles(text, n=self.ngram)
        if not shingles:
            return np.full(self.num_perm, np.iinfo(np.int64).max, dtype=np.int64)

        # Hash each shingle to a 64-bit integer (xxhash if available,
        # else the stdlib blake2b).  Stdlib blake2b is always present
        # and is plenty fast for our purposes.
        shingle_ints = np.fromiter(
            (
                int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
                for s in shingles
            ),
            dtype=np.uint64,
            count=len(shingles),
        )

        # Project each shingle under each permutation: (a*x + b) mod p.
        # Shape: (num_shingles, num_perm) → take min over axis 0.
        a = self._a.astype(np.uint64)
        b = self._b.astype(np.uint64)
        # uint64 wrap-around is intentional; we mod at the end.
        projected = (a[None, :] * shingle_ints[:, None] + b[None, :]) % np.uint64(self._mod)
        sig = projected.min(axis=0).astype(np.int64)
        return sig


class LSHIndex:
    """LSH index over MinHash signatures.

    Splits a signature into ``num_bands`` consecutive rows of
    ``num_rows = num_perm // num_bands`` columns.  Two signatures
    are candidate duplicates iff any band hashes to the same bucket.

    Threshold of duplicate-detection (approx):
        ``1 - (1 - s**num_rows) ** num_bands ≈ 0.5`` at
        ``s ≈ (1/num_bands) ** (1/num_rows)``.

    With ``num_perm=1_000_000`` and ``num_bands=64``,
    ``num_rows=15_625`` and the S-curve is essentially a step
    function: anything below Jaccard 0.85 is *not* a candidate.
    """

    def __init__(self, num_perm: int = 1_000_000, num_bands: int = 64):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.num_rows = num_perm // num_bands
        # band_id -> { bucket_key: [doc_id, ...] }
        self._buckets: list[dict] = [defaultdict(list) for _ in range(num_bands)]

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        assert sig.shape == (self.num_perm,)
        # Reshape to (num_bands, num_rows) and hash each row.
        rows = sig.reshape(self.num_bands, self.num_rows)
        for b in range(self.num_bands):
            # The row bytes are stable; we hash with blake2b to 8 bytes.
            key = int.from_bytes(
                hashlib.blake2b(rows[b].tobytes(), digest_size=8).digest(),
                "little",
            )
            self._buckets[b][key].append(doc_id)

    def candidates(self, doc_id: int, sig: np.ndarray) -> Iterator[int]:
        """Yield candidate duplicate doc-ids for ``doc_id``.

        ``doc_id`` should *not* have been ``add()``-ed already; the
        caller manages the doc-id → text map.
        """
        rows = sig.reshape(self.num_bands, self.num_rows)
        seen: set = set()
        for b in range(self.num_bands):
            key = int.from_bytes(
                hashlib.blake2b(rows[b].tobytes(), digest_size=8).digest(),
                "little",
            )
            for other in self._buckets[b].get(key, ()):
                if other != doc_id and other not in seen:
                    seen.add(other)
                    yield other

    def __len__(self) -> int:
        return sum(len(b) for b in self._buckets)


# ── Top-level deduplicator ─────────────────────────────────────────────────
def _jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Exact Jaccard estimate from two MinHash signatures."""
    return float((sig_a == sig_b).mean())


def deduplicate_docs(
    docs: Sequence[Doc],
    *,
    strategy: str = "minhash",
    num_perm: int = 1_000_000,
    ngram: int = 5,
    num_bands: int = 64,
    prefix_bytes: int = 128,
    min_dup_chars: int = 120,
    jaccard_threshold: float = 0.85,
) -> list[Doc]:
    """Deduplicate ``docs`` using the chosen ``strategy``.

    Selection rule on duplicate clusters: keep the **longest**
    document.  Ties broken by first-seen order.

    Args:
        docs: iterable of ``(text, quality_score)`` tuples.
        strategy: one of ``"minhash"`` (default), ``"prefix"``,
            ``"md5"``.  ``"minhash"`` is the canonical path; the
            other two are smoke-test fallbacks.
        num_perm: MinHash permutations (default 1 M).
        ngram: shingle size in words (default 5).
        num_bands: LSH bands (default 64).  Must divide ``num_perm``.
        prefix_bytes: bytes used by ``"prefix"`` strategy (default 128).
        min_dup_chars: docs shorter than this are skipped from
            MinHash computation (they're not worth the cost; they
            are kept as-is).
        jaccard_threshold: candidate-pair confirmation threshold
            (default 0.85).  Pairs below this are not duplicates.

    Returns:
        Deduplicated list, longest-per-cluster.  Original order
        of the first-seen in each cluster is preserved.
    """
    if strategy == "prefix":
        return exact_prefix_dedup(docs, n=prefix_bytes)
    if strategy == "md5":
        return md5_fallback(docs)

    if strategy != "minhash":
        raise ValueError(f"Unknown dedup strategy: {strategy!r}")

    # ── MinHash path ────────────────────────────────────────────────────
    hasher = MinHashHasher(num_perm=num_perm, ngram=ngram)
    index = LSHIndex(num_perm=num_perm, num_bands=num_bands)

    # First pass: signature + bucket, but defer the candidate
    # confirmation to the second pass (we need the signatures for
    # all docs first to compute exact Jaccard).
    sigs: list[np.ndarray | None] = []
    keep: list[bool] = [True] * len(docs)  # default: keep
    # Cluster id per doc; 0 means "no cluster yet" (we always assign
    # a cluster, with singletons getting their own id).
    cluster: list[int] = list(range(len(docs)))

    # Single doc → trivial pass-through.
    if len(docs) <= 1:
        return list(docs)

    for i, (text, _score) in enumerate(docs):
        if len(text) < min_dup_chars:
            sigs.append(None)
            continue
        sig = hasher.signature(text)
        sigs.append(sig)
        index.add(i, sig)

    # Second pass: for each doc, walk the LSH candidates, confirm
    # with exact Jaccard, and union-find the clusters.
    parent = list(range(len(docs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for j in index.candidates(i, sig):
            if j <= i or sigs[j] is None:
                continue
            j_sim = _jaccard(sig, sigs[j])
            if j_sim >= jaccard_threshold:
                union(i, j)

    # Compute cluster root for each doc.
    for i in range(len(docs)):
        cluster[i] = find(i)

    # Group docs by cluster root.
    clusters: dict = defaultdict(list)
    for i, root in enumerate(cluster):
        clusters[root].append(i)

    # For each cluster, keep the longest doc (ties: first-seen).
    keep_idx: set = set()
    for root, members in clusters.items():
        members_sorted = sorted(
            members,
            key=lambda j: (len(docs[j][0]), -j),  # longer wins, then earlier
            reverse=True,
        )
        keep_idx.add(members_sorted[0])

    return [docs[i] for i in range(len(docs)) if i in keep_idx]
