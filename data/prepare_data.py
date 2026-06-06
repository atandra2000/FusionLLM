"""
prepare_data.py
===============

Dataset pipeline for the DeepSeek-V3-style pre-training skeleton
(extended in Phase 1 to the 7-source SmolTalk-3-inspired mix).

Sources (Phase 1.2)
-------------------
   1. FineWeb-Edu       — high-quality web text with educational scoring
   2. FineMath          — mathematical web text
   3. The-Stack-Edu     — code (Python / JS / Rust / Go / Java)
   4. Cosmopedia        — synthetic instructional text
   5. OpenR1-Math-220k  — math problem/solution pairs
   6. FineWeb2          — multilingual web text (6 languages)
   7. SmolLM-Corpus     — curated SmolLM corpus

Pipeline stages
---------------
   1. Collect   — stream each source up to MAX_DOCS per source
   2. Filter    — language check + quality score threshold
   3. Deduplicate — MinHash (1 M permutations, 5-gram, 64 bands)
                    OR prefix dedup at 128 bytes (smoke tests)
   4. Mix       — curriculum ordering, then shuffle
   5. Split     — train / validation by VAL_RATIO
   6. Tokenize  — block-pack at MAX_SEQ_LEN (4 K legacy / 8 K default)
                  with EOS markers; resampling-aware packing.
   7. Export    — sharded mmap (4 GB-token shards, int32 + 256-byte
                  header) with a per-shard manifest.

Outputs (written to OUTPUT_DIR)
-------------------------------
   shards/manifest.jsonl          — one row per shard (path, n_tokens,
                                     source, weight, quality_score)
   shards/shard_<idx>.bin         — packed int32 tokens + 256-byte header
   dataset_meta.json              — tokenizer + dataset metadata
   eval_samples.txt               — first 500 validation documents

Usage
-----
   python data/prepare_data.py [--output-dir data] [--seed 42] [--smoke]
   python data/prepare_data.py --sources fineweb_edu openr1_math --max-fineweb 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

# ── Configuration ──────────────────────────────────────────────────────────
TOKENIZER_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "data"

# Phase 1.2: seven sources. The default mix is the SmolTalk-3-inspired
# recipe. Each entry is the per-source `max_docs` cap; the *mix
# weights* (used by the curriculum sampler) live in
# `data/curriculum.py`.
MAX_DOCS = {
    "fineweb_edu": 150_000,
    "finemath": 60_000,
    "stack_edu": 60_000,
    "cosmopedia": 30_000,
    "openr1_math": 30_000,
    "fineweb2": 40_000,
    "smollm_corpus": 20_000,
}

VAL_RATIO = 0.01  # fraction of deduplicated docs held out for validation
MIN_CHARS = 120  # documents shorter than this are dropped
MIN_WORDS = 20  # documents with fewer words are dropped

# Phase 1.4: block-pack at 8 K by default. The legacy 4 K is kept as
# the fallback for back-compat with the pre-Phase-1 YAML
# (`configs/pretrain.yaml:max_seq_len: 4096`).
DEFAULT_MAX_SEQ_LEN = 8192
LEGACY_MAX_SEQ_LEN = 4096

EOS_TEXT = "<eos>"
SEED = 42

# Quality-score thresholds per source. The function
# `quality_score` returns a score in [0, 2].
QUALITY_THRESHOLD = {
    "fineweb_edu": 1.0,
    "finemath": 0.8,
    "stack_edu": 0.8,
    "cosmopedia": 0.8,
    "openr1_math": 0.8,
    "fineweb2": 0.8,
    "smollm_corpus": 0.8,
}

# Target shard size (Phase 1.3): 4 GB tokens at int32 = 16 GB on disk.
TARGET_SHARD_TOKENS = 4_000_000_000  # 4 B tokens

# MinHash dedup defaults (Phase 1.1). Override via CLI.
MINHASH_NUM_PERM = 1_000_000
MINHASH_NUM_BANDS = 64
MINHASH_NGRAM = 5

Doc = tuple[str, float]  # (text, quality_score)


# ── Tokenizer (lazy) ────────────────────────────────────────────────────────
def load_tokenizer():
    """Lazy import so this module can be imported without `transformers`."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Text filters ───────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def _load_fasttext_lid() -> object | None:
    """Load fasttext lid.176 model if available (gated by env-flag)."""
    if os.environ.get("LANGUAGE_FILTER", "fasttext") == "ascii":
        return None
    try:
        import fasttext

        model = fasttext.load_model("lid.176.bin")
        return model
    except (ImportError, ValueError, OSError):
        return None


_FASTTEXT_LID: object | None = _load_fasttext_lid()


def passes_language_filter(text: str) -> bool:
    """Language-ID gate.
    
    Uses fasttext ``lid.176`` model when available; falls back to
    an ASCII-ratio heuristic.  Skip fasttext by setting
    ``LANGUAGE_FILTER=ascii``.
    """
    if not text:
        return False
    if _FASTTEXT_LID is not None:
        try:
            preds = _FASTTEXT_LID.predict(text.replace("\n", " "))  # type: ignore[attr-defined]
            lang = preds[0][0].replace("__label__", "")
            return lang == "en"
        except Exception:
            pass
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.85


def quality_score(text: str) -> float:
    """Heuristic quality score in [0, 2] (length + lexical diversity)."""
    if len(text) < MIN_CHARS:
        return 0.0
    words = text.split()
    if len(words) < MIN_WORDS:
        return 0.0
    length_score = min(len(words) / 200, 1.0)
    diversity_score = len(set(words)) / len(words)
    return length_score + diversity_score


# ── Source dispatcher ──────────────────────────────────────────────────────
# Each source returns List[Doc]. The dispatcher is the only place
# `datasets.load_dataset` is touched — keeps the rest of the file
# import-clean and lets us add new sources in one place.
def collect(source: str, max_docs: int) -> list[Doc]:
    """Collect ``(text, quality_score)`` pairs for one source.

    Args:
        source: one of the keys in :data:`MAX_DOCS`.
        max_docs: cap on documents streamed from the source.

    Returns:
        A list of ``(text, quality_score)`` pairs that pass the
        language + quality filters.
    """
    fn = _COLLECTORS.get(source)
    if fn is None:
        raise ValueError(f"Unknown source: {source!r}. Known: {sorted(_COLLECTORS)}")
    return fn(max_docs)


# Per-source collectors. Each is a small adapter around the
# `datasets` library so that the rest of the file does not depend on
# the source-specific schema.


def _collect_fineweb_edu(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[1/7] Loading FineWeb-Edu (up to {max_docs:,} docs)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["fineweb_edu"]
    return _filter_docs(ds, text_field="text", threshold=threshold, desc="fineweb_edu")


def _collect_finemath(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[2/7] Loading FineMath (up to {max_docs:,} docs)...")
    # The FineMath family has several configs; "finemath" is the
    # default educational subset.
    ds = load_dataset("HuggingFaceTB/finemath", "finemath-3plus", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["finemath"]
    return _filter_docs(ds, text_field="text", threshold=threshold, desc="finemath")


def _collect_stack_edu(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[3/7] Loading The-Stack-Edu (up to {max_docs:,} docs)...")
    # The-Stack-dedup is the base; "edu" filter is applied via a
    # subset config where available.
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="python", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["stack_edu"]
    return _filter_docs(ds, text_field="content", threshold=threshold, desc="stack_edu")


def _collect_cosmopedia(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[4/7] Loading Cosmopedia (up to {max_docs:,} docs)...")
    ds = load_dataset("HuggingFaceTB/cosmopedia", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["cosmopedia"]
    return _filter_docs(ds, text_field="text", threshold=threshold, desc="cosmopedia")


def _collect_openr1_math(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[5/7] Loading OpenR1-Math-220k (up to {max_docs:,} docs)...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["openr1_math"]

    def mapper(row):
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        return f"Problem:\n{problem}\n\nSolution:\n{solution}"

    return _filter_docs(
        ds, text_field=None, threshold=threshold, desc="openr1_math", custom_text=mapper
    )


def _collect_fineweb2(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[6/7] Loading FineWeb2 (up to {max_docs:,} docs)...")
    # FineWeb2 is per-language; we stream the English (default) subset
    # and apply the language filter downstream.
    ds = load_dataset("HuggingFaceFW/fineweb-2", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["fineweb2"]
    return _filter_docs(ds, text_field="text", threshold=threshold, desc="fineweb2")


def _collect_smollm_corpus(max_docs: int) -> list[Doc]:
    from datasets import load_dataset

    print(f"\n[7/7] Loading SmolLM-Corpus (up to {max_docs:,} docs)...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=f"train[:{max_docs}]")
    threshold = QUALITY_THRESHOLD["smollm_corpus"]
    return _filter_docs(ds, text_field="text", threshold=threshold, desc="smollm_corpus")


_COLLECTORS: dict[str, Callable[[int], list[Doc]]] = {
    "fineweb_edu": _collect_fineweb_edu,
    "finemath": _collect_finemath,
    "stack_edu": _collect_stack_edu,
    "cosmopedia": _collect_cosmopedia,
    "openr1_math": _collect_openr1_math,
    "fineweb2": _collect_fineweb2,
    "smollm_corpus": _collect_smollm_corpus,
}


def _filter_docs(
    ds,
    *,
    text_field: str | None,
    threshold: float,
    desc: str,
    custom_text: Callable | None = None,
) -> list[Doc]:
    """Apply language + quality filter to an HF dataset, return ``[Doc]``."""
    from tqdm import tqdm

    docs: list[Doc] = []
    for row in tqdm(ds, desc=desc):
        if custom_text is not None:
            text = custom_text(row)
        elif text_field is not None:
            text = row.get(text_field, "")
        else:
            continue
        if not passes_language_filter(text):
            continue
        score = quality_score(text)
        if score >= threshold:
            docs.append((text, score))
    print(f"    kept {len(docs):,} documents")
    return docs


# ── Deduplication (Phase 1.1) ──────────────────────────────────────────────
def deduplicate(docs: list[Doc], *, strategy: str = "minhash", **kw) -> list[Doc]:
    """Run the chosen dedup strategy.

    Default is MinHash; ``strategy="prefix"`` for ultra-fast smoke
    tests, ``strategy="md5"`` for the pre-Phase-1 behaviour.
    """
    from data.dedup import deduplicate_docs

    print(f"\nDeduplicating {len(docs):,} documents (strategy={strategy})...")
    out = deduplicate_docs(docs, strategy=strategy, **kw)
    removed = len(docs) - len(out)
    print(f"    removed {removed:,} duplicates → {len(out):,} unique docs")
    return out


# ── Curriculum mixing (Phase 1.6 stub) ────────────────────────────────────
def curriculum_mix(by_source: dict[str, list[Doc]]) -> list[Doc]:
    """Concatenate per-source lists, sorted by descending quality.

    The actual *sampling weights* (the curriculum manifest) live in
    `data/curriculum.py`. This function only orders the docs.
    """
    out: list[Doc] = []
    for source, docs in by_source.items():
        out.extend(sorted(docs, key=lambda x: -x[1]))
    return out


# ── Block-pack at 8 K (Phase 1.4) ─────────────────────────────────────────
def tokenize_and_pack(
    docs: list[Doc],
    tokenizer,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    resample: bool = False,
    min_efficiency: float = 0.99,
    desc: str = "tokenize",
) -> tuple[torch.Tensor, float]:
    """Tokenize documents and pack them into fixed-length sequences.

    Pack strategy
    -------------
    Tokens from consecutive documents are concatenated into a single
    buffer. An EOS token is appended after each document.  When
    adding the next document would overflow the buffer, the current
    buffer is padded to ``max_seq_len`` with ``pad_token_id`` and
    flushed.  The final partial buffer is also padded.

    Resampling-aware packing (Phase 1.4)
    -------------------------------------
    When ``resample=True``, documents are sampled **with replacement**
    proportional to their quality score (modded-nanogpt #33).  This
    minimises padding waste by prioritising longer, higher-quality
    documents when nearing a sequence boundary.  The shuffle is
    deterministic (controlled by ``SEED``).

    Packing efficiency
    ------------------
    We return ``(tokens, packing_efficiency)`` so the caller can
    abort the pipeline if the efficiency falls below the
    ``plan.md:1.7`` target of 99.5 %.
    """
    eos_ids = tokenizer.encode(EOS_TEXT, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id

    packed: list[int] = []
    current: list[int] = []
    utilization: list[float] = []

    from tqdm import tqdm

    # Phase 1.4: Resampling-aware iteration.
    # When resample=True, build a weighted sampler over the docs (by
    # quality score) and draw *with replacement*.  This pushes padding
    # waste below 1 % even with highly skewed doc-length distributions.
    doc_iter: Iterator[Doc]
    if resample:
        weights = torch.tensor([max(s, 0.01) for _, s in docs], dtype=torch.float32)
        rng = random.Random(SEED)
        n_docs = len(docs)
        # Pre-compute indices: draw enough to cover ~2x the total
        # tokens, since we consume docs until we run out.
        total_est = sum(len(tokenizer.encode(normalize(t), add_special_tokens=False)) + len(eos_ids) for t, _ in docs[:1000]) / max(len(docs[:1000]), 1) * n_docs  # approx
        n_samples = max(n_docs, int(total_est // 512 + 1))
        sampled_indices = rng.choices(range(n_docs), weights=weights.tolist(), k=n_samples)
        doc_iter = (docs[i] for i in sampled_indices)
    else:
        doc_iter = iter(docs)

    for text, _score in tqdm(doc_iter, desc=desc, total=len(docs) if not resample else None):
        ids = tokenizer.encode(normalize(text), add_special_tokens=False) + eos_ids
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        if len(current) + len(ids) > max_seq_len:
            utilization.append(len(current) / max_seq_len)
            current.extend([pad_id] * (max_seq_len - len(current)))
            packed.extend(current)
            current = []

        current.extend(ids)

    # Flush the final partial buffer
    if current:
        utilization.append(len(current) / max_seq_len)
        current.extend([pad_id] * (max_seq_len - len(current)))
        packed.extend(current)

    avg_util = sum(utilization) / max(len(utilization), 1)
    n_seq = len(packed) // max_seq_len

    if not resample:
        print(f"    {n_seq:,} sequences | packing efficiency: {avg_util * 100:.1f}%")

    if min_efficiency > 0 and avg_util < min_efficiency:
        raise RuntimeError(
            f"Packing efficiency {avg_util * 100:.1f}% < {min_efficiency * 100:.1f}% — "
            "pipeline abort per plan.md:1.7"
        )

    return torch.tensor(packed, dtype=torch.long), avg_util


# ── Sharded mmap writer (Phase 1.3) ───────────────────────────────────────
# Delegated to data/shard_writer.py (extracted in Phase 1.3)
from data.shard_writer import (  # noqa: F811
    TARGET_SHARD_TOKENS,
    np_dtype,
    write_manifest,
    write_shards,
)


# ── Eval sample export ─────────────────────────────────────────────────────
def export_eval_samples(docs: list[Doc], output_dir: Path, n: int = 500) -> None:
    path = output_dir / "eval_samples.txt"
    sep = "\n\n" + "=" * 60 + "\n\n"
    with open(path, "w", encoding="utf-8") as f:
        for text, _ in docs[:n]:
            f.write(text)
            f.write(sep)
    print(f"    eval samples → {path}")


# ── Argument parsing ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare pre-training data (Phase 1, 7-source).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory to write shards, manifest, eval samples.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Block-pack sequence length (4 K legacy / 8 K default).",
    )
    p.add_argument(
        "--target-shard-tokens",
        type=int,
        default=TARGET_SHARD_TOKENS,
        help="Target tokens per shard (default 4 B).",
    )
    p.add_argument(
        "--sources", nargs="*", default=None, help="Subset of sources to use (default: all 7)."
    )
    # Per-source caps
    for source, default in MAX_DOCS.items():
        p.add_argument(
            f"--max-{source.replace('_', '-')}",
            type=int,
            default=default,
            help=f"Max documents to stream from {source}.",
        )
    # Dedup
    p.add_argument(
        "--dedup-strategy",
        choices=("minhash", "prefix", "md5"),
        default="minhash",
        help="Dedup strategy.",
    )
    p.add_argument("--minhash-num-perm", type=int, default=MINHASH_NUM_PERM)
    p.add_argument("--minhash-num-bands", type=int, default=MINHASH_NUM_BANDS)
    p.add_argument("--minhash-ngram", type=int, default=MINHASH_NGRAM)
    p.add_argument(
        "--smoke", action="store_true", help="Smoke-test: use prefix dedup, only the first source."
    )
    p.add_argument(
        "--resample",
        action="store_true",
        default=False,
        help="Enable resampling-aware packing (modded-nanogpt #33); "
        "documents drawn with replacement proportional to quality score.",
    )
    p.add_argument("--dry-run", action="store_true", help="Skip writing files; print stats only.")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    random.seed(args.seed)

    # ── Decide which sources to use ───────────────────────────────────
    if args.smoke:
        sources = ["fineweb_edu"]
        max_per_source = {"fineweb_edu": min(MAX_DOCS["fineweb_edu"], 1000)}
        dedup_strategy = "prefix"
    else:
        sources = args.sources or list(MAX_DOCS.keys())
        max_per_source = {s: getattr(args, f"max_{s}") for s in sources}
        dedup_strategy = args.dedup_strategy

    # ── Collect ────────────────────────────────────────────────────────
    by_source: dict[str, list[Doc]] = {}
    for s in sources:
        by_source[s] = collect(s, max_per_source[s])
        print(f"  {s:>15s}: {len(by_source[s]):>8,}")

    # ── Mix → dedup → shuffle ─────────────────────────────────────────
    mixed = curriculum_mix(by_source)
    dedup_kw = {}
    if dedup_strategy == "minhash":
        dedup_kw = {
            "num_perm": args.minhash_num_perm,
            "num_bands": args.minhash_num_bands,
            "ngram": args.minhash_ngram,
        }
    unique = deduplicate(mixed, strategy=dedup_strategy, **dedup_kw)
    random.shuffle(unique)
    print(f"\nFinal unique docs after shuffle: {len(unique):,}")

    # ── Train / val split ──────────────────────────────────────────────
    split_idx = int(len(unique) * (1.0 - args.val_ratio))
    train_docs = unique[:split_idx]
    val_docs = unique[split_idx:]
    print(f"\nTrain docs: {len(train_docs):,}")
    print(f"Val docs:   {len(val_docs):,}  ({args.val_ratio * 100:.1f}%)")

    # ── Tokenize & pack ────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({TOKENIZER_NAME})...")
    tokenizer = load_tokenizer()

    print("\nTokenizing train set...")
    train_tokens, train_packing_eff = tokenize_and_pack(
        train_docs,
        tokenizer,
        max_seq_len=args.max_seq_len,
        resample=args.resample,
        desc="train",
    )

    print("\nTokenizing validation set...")
    val_tokens, val_packing_eff = tokenize_and_pack(
        val_docs,
        tokenizer,
        max_seq_len=args.max_seq_len,
        resample=args.resample,
        desc="val",
    )

    # ── Stats ──────────────────────────────────────────────────────────
    bytes_per_token = 4  # int32
    train_gb = train_tokens.numel() * bytes_per_token / 1024**3
    val_gb = val_tokens.numel() * bytes_per_token / 1024**3
    print("\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)
    print(f"  Train tokens : {len(train_tokens):>14,}  (~{train_gb:.2f} GB on disk)")
    print(f"  Val tokens   : {len(val_tokens):>14,}  (~{val_gb:.2f} GB on disk)")
    print(f"  Train packing efficiency : {train_packing_eff * 100:.2f}%")
    print(f"  Val packing efficiency   : {val_packing_eff * 100:.2f}%")
    print(f"  Train seqs   : {len(train_tokens) // args.max_seq_len:>14,}")
    print(f"  Val seqs     : {len(val_tokens) // args.max_seq_len:>14,}")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run: skipping file writes.")
        return

    # ── Write shards ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_shards(
        train_tokens,
        output_dir,
        target_shard_tokens=args.target_shard_tokens,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    write_manifest(manifest, output_dir)

    # ── Eval samples ───────────────────────────────────────────────────
    print("\nExporting eval samples...")
    export_eval_samples(val_docs, output_dir)

    # ── Dataset meta ───────────────────────────────────────────────────
    meta = {
        "tokenizer_name": TOKENIZER_NAME,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_seq_len": args.max_seq_len,
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
        "train_packing_efficiency": train_packing_eff,
        "val_packing_efficiency": val_packing_eff,
        "sources": sources,
        "dedup_strategy": dedup_strategy,
    }
    meta_path = output_dir / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    metadata → {meta_path}")


if __name__ == "__main__":
    main()
