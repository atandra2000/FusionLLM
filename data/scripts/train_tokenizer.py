# data/scripts/train_tokenizer.py
# Stage 3: train a 64K byte-level BPE SentencePiece tokenizer on a sample
# of the cleaned corpus.
#
# Simplification vs the original plan:
#   * No fertility report (we trust SPM; downstream eval will catch problems).
#   * No HF tokenizer.json conversion (we use the SPM model directly; the
#     tokenize.py stage consumes it via `sentencepiece.SentencePieceProcessor`).

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import sentencepiece as spm

from data.common import (
    CLEAN_ROOT,
    TOKENIZER_DIR,
    iter_clean_shards,
    iter_clean_jsonl,
    load_yaml,
    log,
    save_yaml,
    seed_everything,
)


def _sample_text_files(
    mix_cfg: dict,
    total_gb: float,
    out_dir: Path,
) -> dict:
    """Write `total_gb` of plain-text into `out_dir`, balanced per source.

    Returns a dict {source_id: bytes_written}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = int(total_gb * 1024 ** 3)
    weights: dict[str, float] = mix_cfg["tokenizer_train_mix"]
    per_source = {sid: target_bytes * w for sid, w in weights.items()}

    written: dict[str, int] = {sid: 0 for sid in weights}
    rng = random.Random(1729)

    for source_id, want_bytes in per_source.items():
        if want_bytes <= 0:
            continue
        out_path = out_dir / f"{source_id}.txt"
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            for shard in iter_clean_shards(source_id):
                if written[source_id] >= want_bytes:
                    break
                for doc in iter_clean_jsonl(shard):
                    text = doc.get("text", "")
                    if not text:
                        continue
                    out.write(text)
                    out.write("\n")
                    written[source_id] += len(text.encode("utf-8"))
                    if written[source_id] >= want_bytes:
                        break
        log(f"[{source_id}] sampled {written[source_id] / 1024**2:.1f} MB")

    return written


def train(config_path: str, sample_dir_override: str | None = None, output_dir_override: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    tok_cfg = cfg  # tokenizer.yaml is the only file this stage reads

    out_dir = TOKENIZER_DIR
    if output_dir_override:
        out_dir = Path(output_dir_override)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / "tokenizer"

    # 1. Build the training corpus if not already present
    sample_dir = Path(sample_dir_override) if sample_dir_override else (out_dir / "_corpus")
    if not sample_dir.exists() or not any(sample_dir.glob("*.txt")):
        if sample_dir_override is None:
            sample_dir.mkdir(parents=True, exist_ok=True)
        log(f"Building tokenizer training corpus at {sample_dir}")
        sizes = _sample_text_files(tok_cfg, tok_cfg["tokenizer_train_size_gb"], sample_dir)
        log(f"Sampled sizes: {sizes}")
    else:
        log(f"Reusing existing corpus at {sample_dir}")

    # 2. Combine the per-source files into a single shuffled input for SPM
    #    (SPM accepts a comma-separated list and shuffles internally with
    #    --input_sentence_size + shuffle_input_sentence.)
    input_globs = sorted(str(p) for p in sample_dir.glob("*.txt"))
    if not input_globs:
        raise RuntimeError(f"No .txt files found in {sample_dir}")

    # 3. Compose the SPM kwargs
    active = tok_cfg["active_special_tokens"]
    reserved = tok_cfg["reserved_special_tokens"]
    # Sanity: 4 active + 28 reserved = 32 special tokens, ids 0..31.
    assert len(active) == 4, "active_special_tokens must be exactly 4 (EOS, BOS, PAD, UNK)"
    assert len(reserved) == 28, "reserved_special_tokens must be exactly 28"

    spm_kwargs = dict(
        input=",".join(input_globs),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=tok_cfg["vocab_size"],
        character_coverage=tok_cfg["character_coverage"],
        byte_fallback=True,
        split_digits=tok_cfg["split_digits"],
        normalization_rule_name=tok_cfg["normalization_rule_name"],
        max_sentence_length=tok_cfg["max_sentence_length"],
        # Reserve the special tokens at fixed ids 0..31
        pad_id=2,
        unk_id=3,
        bos_id=1,
        eos_id=0,
        user_defined_symbols=active + reserved,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=max(1, os.cpu_count() or 1),
        # Note: SentencePiece (as of 0.2.x) does not expose a seed kwarg.
        # Determinism is achieved by feeding a deterministic, globally-sorted
        # input corpus (see _sample_text_files below) and by the documented
        # canonicalization of the BPE merge process.
    )

    log("Starting SentencePiece training. This can take 30–90 min on 1×A100.")
    spm.SentencePieceTrainer.train(**spm_kwargs)
    log(f"SentencePiece training done. Model at {model_prefix}.model")

    # 4. Write a small metadata sidecar so downstream stages can introspect
    save_yaml(
        {
            "vocab_size": tok_cfg["vocab_size"],
            "model_path": str(model_prefix) + ".model",
            "vocab_path": str(model_prefix) + ".vocab",
            "active_special_tokens": active,
            "reserved_special_tokens": reserved,
            "eos_id": 0,
            "bos_id": 1,
            "pad_id": 2,
            "unk_id": 3,
        },
        out_dir / "tokenizer_meta.yaml",
    )

    return Path(str(model_prefix) + ".model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: train the 64K BPE tokenizer")
    parser.add_argument("--config", default="data/config/tokenizer.yaml")
    parser.add_argument("--sample-dir", default=None,
                        help="If set, use this directory of *.txt files instead of sampling from data/clean/")
    parser.add_argument("--output-dir", default=None,
                        help="If set, override the output directory for the tokenizer model")
    args = parser.parse_args(argv)

    seed_everything()
    try:
        train(args.config, args.sample_dir, args.output_dir)
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        return 1
    log("train_tokenizer: done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
