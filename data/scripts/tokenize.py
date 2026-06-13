# data/scripts/tokenize.py
# Stage 4: tokenize data/clean/<source>/shard_*.jsonl.zst into
# data/tokens/<source>/tokens_*.bin (uint16 LE).
#
# Output of this stage is a per-source, sequential stream of uint16 token
# IDs, with each document terminated by EOS_ID (0). The shard_writer stage
# reads these files in deterministic order and packs them into
# seq_len=4096 sequences.
#
# Simplification vs the original plan:
#   * No global MinHash dedup (the corpus is large enough that small
#     duplication is acceptable; we'd rather spend the time training).
#   * No intra-shard simhash.
#   * Single-process, ordered by source. Parallelism lives in shard_writer.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

from data.common import (
    EOS_ID,
    TOKENIZER_DIR,
    TOKENS_ROOT,
    iter_clean_jsonl,
    iter_clean_shards,
    load_state,
    load_yaml,
    log,
    save_state,
    seed_everything,
)

# Cap on tokens per output .bin file (~512 MB at uint16 = 256 M tokens)
TOKENS_PER_FILE = 256_000_000


def _open_spm() -> spm.SentencePieceProcessor:
    model_path = TOKENIZER_DIR / "tokenizer.model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {model_path}. Run train_tokenizer.py first."
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


def _tokenize_source(
    source_id: str,
    sp: spm.SentencePieceProcessor,
    state: dict,
) -> int:
    """Tokenize one source. Returns the number of tokens written."""
    in_files = list(iter_clean_shards(source_id))
    if not in_files:
        log(f"[{source_id}] no clean files; skipping")
        return 0

    out_dir = TOKENS_ROOT / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_state = state.setdefault(source_id, {})
    out_idx = int(src_state.get("out_idx", 0))
    in_idx = int(src_state.get("in_idx", 0))
    in_doc = int(src_state.get("in_doc", 0))
    tokens_written = int(src_state.get("tokens_written", 0))

    out_path = out_dir / f"tokens_{out_idx:05d}.bin"
    out_f = open(out_path, "wb")
    buf = np.empty(TOKENS_PER_FILE, dtype=np.uint16)
    buf_pos = 0
    total_new = 0

    # Buffers for the encode call: SPM works in pieces, but a single encode
    # call is faster than many small ones for our throughput regime.
    log(f"[{source_id}] starting in_idx={in_idx} in_doc={in_doc} out_idx={out_idx}")

    for fi in range(in_idx, len(in_files)):
        in_path = in_files[fi]
        for doc_idx, doc in enumerate(iter_clean_jsonl(in_path)):
            if fi == in_idx and doc_idx < in_doc:
                continue
            text = doc.get("text", "")
            if not text:
                continue

            ids = sp.EncodeAsIds(text)
            if not ids:
                continue
            # Append EOS
            ids.append(EOS_ID)

            n = len(ids)
            # If this doc is bigger than the file buffer, flush what we have
            # and write the doc straight from a contiguous numpy array.
            if n > TOKENS_PER_FILE:
                if buf_pos > 0:
                    out_f.write(buf[:buf_pos].tobytes())
                    buf_pos = 0
                arr = np.asarray(ids, dtype=np.uint16)
                out_f.write(arr.tobytes())
                tokens_written += n
                total_new += n
            else:
                if buf_pos + n > TOKENS_PER_FILE:
                    out_f.write(buf[:buf_pos].tobytes())
                    out_f.close()
                    out_idx += 1
                    out_path = out_dir / f"tokens_{out_idx:05d}.bin"
                    out_f = open(out_path, "wb")
                    buf_pos = 0
                buf[buf_pos:buf_pos + n] = ids
                buf_pos += n
                tokens_written += n
                total_new += n

            if total_new % 5_000_000 < n:  # save state ~every 5M tokens
                src_state.update(
                    out_idx=out_idx,
                    in_idx=fi,
                    in_doc=doc_idx + 1,
                    tokens_written=tokens_written,
                )
                save_state("tokenize", state)

    if buf_pos > 0:
        out_f.write(buf[:buf_pos].tobytes())
    out_f.close()

    # If the last file is empty, delete it
    if buf_pos == 0 and out_path.exists() and out_idx > 0:
        out_path.unlink()
        out_idx -= 1

    src_state.update(
        out_idx=out_idx + 1,
        in_idx=len(in_files),
        in_doc=0,
        tokens_written=tokens_written,
    )
    save_state("tokenize", state)
    log(f"[{source_id}] wrote {total_new:,} tokens across {out_idx + 1} file(s) in {out_dir}")
    return total_new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4: tokenize clean jsonl.zst -> uint16 token files")
    parser.add_argument("--config", default="data/config/mixture.yaml")
    parser.add_argument("--source", default=None)
    args = parser.parse_args(argv)

    seed_everything()
    mixture = load_yaml(args.config)
    sp = _open_spm()
    state = load_state("tokenize")

    # Quick sanity check on the tokenizer
    if sp.GetPieceSize() != 64000:
        log(f"WARNING: tokenizer vocab size is {sp.GetPieceSize()}, expected 64000")

    for spec in mixture["sources"]:
        if args.source and spec["id"] != args.source:
            continue
        try:
            _tokenize_source(spec["id"], sp, state)
        except KeyboardInterrupt:
            log("Interrupted. State saved; rerun to resume.")
            return 130
        except Exception as e:
            log(f"[{spec['id']}] ERROR: {type(e).__name__}: {e}")
            save_state("tokenize", state)
            return 1

    log("tokenize: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
