# data/scripts/preprocess.py

from __future__ import annotations

import argparse
import re
import sys

from data.common import (
    CLEAN_ROOT,
    RAW_ROOT,
    iter_clean_jsonl,
    iter_source_shards,
    json_escape,
    light_clean,
    load_state,
    load_yaml,
    log,
    open_text_writer,
    progress,
    save_state,
    seed_everything,
)

MIN_CHARS = 200
MAX_CHARS = 200_000
MAX_SYMBOL_RATIO = 0.20
MAX_URL_DENSITY = 0.30

_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+")
_WORD_RE = re.compile(r"\b[\w']+\b")
_NONALNUM_RE = re.compile(r"[^A-Za-z0-9\s]")

ROTATE_BYTES = 256 * 1024 * 1024


def _passes_filters(text: str) -> bool:
    n = len(text)
    if n < MIN_CHARS or n > MAX_CHARS:
        return False
    nonalnum = len(_NONALNUM_RE.findall(text))
    if nonalnum / max(n, 1) > MAX_SYMBOL_RATIO:
        return False
    words = _WORD_RE.findall(text)
    if not words:
        return False
    urls = _URL_RE.findall(text)
    if len(urls) / len(words) > MAX_URL_DENSITY:
        return False
    return True


def preprocess_source(source_id: str, state: dict, *, in_start: int | None = None) -> int:
    in_dir = RAW_ROOT / source_id
    out_dir = CLEAN_ROOT / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_state = state.setdefault(source_id, {})
    out_file_idx = int(src_state.get("out_file_idx", 0))
    rows_in = int(src_state.get("rows_in", 0))
    rows_out = int(src_state.get("rows_out", 0))
    cur_in_idx = int(src_state.get("cur_in_idx", 0))
    cur_in_row = int(src_state.get("cur_in_row", 0)) if in_start is None else int(in_start)

    raw_files = list(iter_source_shards(RAW_ROOT, source_id))
    if not raw_files:
        log(f"[{source_id}] no raw files found")
        return 0

    out_path = out_dir / f"shard_{out_file_idx:05d}.jsonl.zst"
    writer = open_text_writer(out_path)
    bytes_in_file = 0
    new_in = 0
    new_out = 0

    log(f"[{source_id}] starting at in_file={cur_in_idx} in_row={cur_in_row} out_file={out_file_idx}")

    for fi in range(cur_in_idx, len(raw_files)):
        in_path = raw_files[fi]
        for row in iter_clean_jsonl(in_path):
            if fi == cur_in_idx and new_in < cur_in_row:
                new_in += 1
                continue

            text = row.get("text", "")
            if not text:
                new_in += 1
                continue

            text = light_clean(text)
            if not _passes_filters(text):
                new_in += 1
                continue

            line = ('{"text": ' + json_escape(text) + "}\n").encode("utf-8")
            writer.write(line)
            bytes_in_file += len(line)
            new_out += 1
            new_in += 1
            rows_in += 1
            rows_out += 1

            if bytes_in_file >= ROTATE_BYTES:
                writer.close()
                out_file_idx += 1
                out_path = out_dir / f"shard_{out_file_idx:05d}.jsonl.zst"
                writer = open_text_writer(out_path)
                bytes_in_file = 0

            if new_out % 50_000 == 0 and new_out > 0:
                src_state.update(
                    cur_in_idx=fi,
                    cur_in_row=new_in if fi == cur_in_idx else 0,
                    out_file_idx=out_file_idx,
                    rows_in=rows_in,
                    rows_out=rows_out,
                )
                save_state("preprocess", state)

        cur_in_idx = fi + 1
        cur_in_row = 0

    writer.close()
    if bytes_in_file == 0 and out_path.exists():
        out_path.unlink()
        out_file_idx = max(0, out_file_idx - 1)

    src_state.update(
        cur_in_idx=cur_in_idx,
        cur_in_row=0,
        out_file_idx=out_file_idx + 1,
        rows_in=rows_in,
        rows_out=rows_out,
    )
    save_state("preprocess", state)
    log(f"[{source_id}] kept {new_out} / {new_in} rows")
    return new_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2: clean raw jsonl.zst")
    parser.add_argument("--config", default="data/config/mixture.yaml")
    parser.add_argument("--source", default=None, help="If set, only process this source id")
    args = parser.parse_args(argv)

    seed_everything()
    mixture = load_yaml(args.config)
    state = load_state("preprocess")

    for spec in mixture["sources"]:
        if args.source and spec["id"] != args.source:
            continue
        try:
            preprocess_source(spec["id"], state)
        except KeyboardInterrupt:
            log("Interrupted. State saved.")
            return 130
        except Exception as e:
            log(f"[{spec['id']}] ERROR: {type(e).__name__}: {e}")
            save_state("preprocess", state)
            return 1

    log("preprocess: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
