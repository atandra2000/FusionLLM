# data/scripts/download_raw.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from data.common import (
    RAW_ROOT,
    SourceBudget,
    compute_source_budgets,
    json_escape,
    light_clean,
    load_state,
    load_yaml,
    log,
    progress,
    save_state,
    seed_everything,
    source_to_spec,
    open_text_writer,
)

_DOLMA_SUBSET_FILTER = {
    "wikipedia": "wikipedia",
    "books": "books",
}

ROTATE_BYTES = 256 * 1024 * 1022


def _iter_source(spec: dict) -> Iterator[dict]:
    ds = load_dataset(
        spec["hf_id"],
        spec["hf_config"],
        split=spec["hf_split"],
        streaming=True,
        revision=spec.get("hf_revision", "main"),
        trust_remote_code=False,
    )
    text_field = spec.get("text_field", "text")
    subset = _DOLMA_SUBSET_FILTER.get(spec["id"])
    for row in ds:
        if subset is not None and row.get("source") != subset:
            continue
        if text_field not in row or not row[text_field]:
            continue
        yield {"text": row[text_field]}


def download_source(
    spec: dict,
    budget: SourceBudget,
    state: dict,
    *,
    max_docs: int | None = None,
) -> int:
    out_dir = RAW_ROOT / spec["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    src_state = state.setdefault(spec["id"], {})
    files_written = int(src_state.get("files_written", 0))
    rows_emitted = int(src_state.get("rows_emitted", 0))
    target_docs = (budget.target_train_tokens + budget.target_val_tokens + budget.target_test_tokens) // 100

    if rows_emitted >= target_docs:
        log(f"[{spec['id']}] already have {rows_emitted} rows, skipping")
        return 0

    log(f"[{spec['id']}] streaming {spec['hf_id']} / {spec['hf_config']}")
    iterator = _iter_source(spec)

    skipped = 0
    while skipped < rows_emitted:
        try:
            next(iterator)
            skipped += 1
        except StopIteration:
            log(f"[{spec['id']}] WARNING: stream exhausted during skip")
            return 0

    file_idx = files_written
    out_path = out_dir / f"shard_{file_idx:05d}.jsonl.zst"
    writer = open_text_writer(out_path)
    bytes_in_file = 0
    file_rows = 0
    new_rows = 0

    for row in progress(iterator, desc=f"[{spec['id']}]"):
        text = light_clean(row["text"])
        if not text:
            continue

        line = ('{"text": ' + json_escape(text) + "}\n").encode("utf-8")
        writer.write(line)
        bytes_in_file += len(line)
        file_rows += 1
        new_rows += 1
        rows_emitted += 1

        if bytes_in_file >= ROTATE_BYTES:
            writer.close()
            file_idx += 1
            out_path = out_dir / f"shard_{file_idx:05d}.jsonl.zst"
            writer = open_text_writer(out_path)
            bytes_in_file = 0
            file_rows = 0

        if new_rows % 10_000 == 0:
            src_state["files_written"] = file_idx
            src_state["rows_emitted"] = rows_emitted
            save_state("download_raw", state)

        if max_docs is not None and new_rows >= max_docs:
            break

    writer.close()
    if file_rows == 0 and out_path.exists():
        out_path.unlink()
        file_idx -= 1

    src_state["files_written"] = file_idx + 1
    src_state["rows_emitted"] = rows_emitted
    save_state("download_raw", state)

    log(f"[{spec['id']}] wrote {new_rows} rows across {file_idx + 1} file(s)")
    return new_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1: stream HF datasets into raw jsonl.zst files")
    parser.add_argument("--config", default="data/config/mixture.yaml")
    parser.add_argument("--source", default=None, help="If set, only process this source id")
    parser.add_argument("--max-docs", type=int, default=None, help="Cap each source at this many rows")
    args = parser.parse_args(argv)

    seed_everything()
    mixture = load_yaml(args.config)
    budgets = {b.source_id: b for b in compute_source_budgets(mixture)}
    state = load_state("download_raw")

    for spec in mixture["sources"]:
        if args.source and spec["id"] != args.source:
            continue
        budget = budgets[spec["id"]]
        try:
            download_source(spec, budget, state, max_docs=args.max_docs)
        except KeyboardInterrupt:
            log("Interrupted. State has been saved.")
            return 130
        except Exception as e:
            log(f"[{spec['id']}] ERROR: {type(e).__name__}: {e}")
            save_state("download_raw", state)
            return 1

    log("download_raw: all requested sources done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
