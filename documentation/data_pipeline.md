# Data Pipeline — 6 stages, 64K BPE, source mix

> Source: `data/` (`common.py`, `prepare_data.py`, `scripts/`)

FusionLLM's data pipeline transforms raw HuggingFace streaming datasets
into memory-mapped `.npy` shards ready for training. It is a thin
project-specific shim (`data/prepare_data.py`) over a shared universal
pipeline; the only FusionLLM-specific part is the **custom 64K BPE
tokenizer** (vocab 64,000, EOS=0, BOS=1, PAD=2).

## The 6 stages

| # | Script | Input → Output | Description |
|---|--------|----------------|-------------|
| 1 | `download_raw.py` | HF streaming datasets → `data/raw/*/shard_*.jsonl.zst` | Streams HuggingFace datasets, applies `light_clean`, writes rotated zstd-compressed JSONL shards (256 MB per file). |
| 2 | `preprocess.py` | Raw → `data/clean/*/shard_*.jsonl.zst` | Applies quality filters (min/max chars, symbol ratio, URL density) and `light_clean`, writes cleaned JSONL. |
| 3 | `train_tokenizer.py` | Clean sample → `data/tokenizer/tokenizer.model` | Trains a 64K byte-level BPE tokenizer with SentencePiece (byte fallback, digit split, NFKC). |
| 4 | `tokenize.py` | Clean → `data/tokens/*/tokens_*.bin` | Encodes text to `uint16` token sequences with EOS termination, writes 256 M-token files. |
| 5 | `shard_writer.py` | Tokens → `data/shards/{train,val,test}/shard_*.npy` | Packs tokens into 4096×4096 `int32` memory-mapped shards (EOS-padded to `MAX_SEQ_LEN`). Writes `manifest.json`. |
| 6 | `streaming_dataloader.py` | Shards → `(tokens, targets)` tensors | Memory-maps shards, shuffles sequences, yields `(B, T)` `long` tensors with `targets = roll(tokens, -1)` (last target = EOS). |

Each stage is **resumable**: progress is tracked via JSON state files in
`data/state/<stage>.json`, written atomically (`.tmp` → replace). A
`KeyboardInterrupt` saves state and exits 130; an exception saves state
and exits 1.

## Source mix (8.31B tokens)

The mixture is defined in `data/config/mixture.yaml`. The 6 sources and
their weights:

| Dataset | Weight | Approx. tokens | HF id |
|---------|:-----:|:--------------:|-------|
| FineWeb-Edu | 55% | 4.57B | `HuggingFaceFW/fineweb-edu` (sample-10BT) |
| FineWeb | 20% | 1.66B | `HuggingFaceFW/fineweb` (sample-10BT) |
| The Stack v2 (Python) | 10% | 0.83B | `bigcode/the-stack-v2-train-full-ids` |
| SlimPajama | 8% | 0.66B | `cerebras/SlimPajama-627B` (dedup) |
| Dolma Wikipedia | 4% | 0.33B | `allenai/dolma` (v1_6-sample), subset=wikipedia |
| Dolma Books | 3% | 0.25B | `allenai/dolma` (v1_6-sample), subset=books |

**Splits:** train 97% (8.056B), validation 1.5% (124.65M), test 1.5%
(124.65M).

> **SKILLS.md Skill 4 pitfalls:** mixture weights must sum to 1.0 ± 1e-6
> (the loader asserts). All sources must tokenize with the same 64K BPE
> tokenizer — different tokenizers cause silent vocab-id collisions.

## Shared utilities (`data/common.py`)

- **Special tokens:** `EOS_ID=0`, `BOS_ID=1`, `PAD_ID=2`, `UNK_ID=3`,
  reserved 4–31. `VOCAB_SIZE=64_000`, `MAX_SEQ_LEN=4096`.
- **Shard dtype:** `np.int32` (`SHARD_DTYPE`).
- **Paths:** `DATA_ROOT`, `RAW_ROOT`, `CLEAN_ROOT`, `TOKENS_ROOT`,
  `SHARDS_ROOT`, `TOKENIZER_DIR`, `STATE_ROOT`, `LOGS_ROOT`,
  `CONFIG_ROOT`.
- **I/O:** `open_text_writer` / `open_text_reader` (zstd stream
  compression), `load_yaml` / `save_yaml`, `load_state` / `save_state`
  / `clear_state` (atomic JSON state), `log` (console + `pipeline.log`).
- **Hashing:** `stable_hash_u64` (FNV-1a over NFC-normalized first 8 KB).
- **Cleaning:** `light_clean` — NFKC normalize, strip control chars /
  email / IPv4 / phone PII, collapse whitespace.
- **Budgeting:** `compute_source_budgets(mixture)` splits the total
  token budget across sources by weight and across train/val/test by
  fraction. `SourceBudget` holds the per-source targets.

## Stage 1 — `download_raw.py`

- Streams each source via `datasets.load_dataset(..., streaming=True,
  trust_remote_code=False)`.
- Dolma sources are filtered by `row["source"] == subset` (wikipedia /
  books) via `_DOLMA_SUBSET_FILTER`.
- Each row's `text` field is `light_clean`-ed and written as
  `{"text": "..."}\n` (zstd-compressed). Files rotate at 256 MB
  (`ROTATE_BYTES`).
- State tracks `files_written` and `rows_emitted` per source; on resume,
  already-emitted rows are skipped by draining the iterator.

## Stage 2 — `preprocess.py`

- Quality filters (`_passes_filters`): `200 ≤ len(text) ≤ 200_000`,
  symbol ratio `≤ 0.20`, URL density `≤ 0.30`, must contain ≥1 word.
- Output is `light_clean`-ed again and written to `data/clean/<source>/`.
- State tracks `cur_in_idx` / `cur_in_row` / `out_file_idx` / `rows_in`
  / `rows_out` per source.

## Stage 3 — `train_tokenizer.py`

- Builds a tokenizer training corpus by sampling `tokenizer_train_size_gb`
  of clean text per source, weighted by `tokenizer_train_mix`.
- Trains a SentencePiece BPE: `vocab_size` (64K for the real run),
  `byte_fallback=True`, `split_digits`, NFKC normalization,
  `character_coverage`, `pad_id=2`, `unk_id=3`, `bos_id=1`, `eos_id=0`,
  plus 4 active + 28 reserved special tokens.
- Writes `tokenizer.model`, `tokenizer.vocab`, and
  `tokenizer_meta.yaml`.

## Stage 4 — `tokenize.py`

- Loads the trained SentencePiece model and encodes each clean document
  to `uint16` token IDs, appending `EOS_ID` after each document.
- Writes `tokens_*.bin` files of `TOKENS_PER_FILE = 256_000_000` tokens
  each. Documents larger than one file are flushed and written directly.
- State tracks `out_idx` / `in_idx` / `in_doc` / `tokens_written`.

## Stage 5 — `shard_writer.py`

- Streams each source's tokens by splitting on `EOS_ID` boundaries
  (`_stream_source_tokens`), yielding one document array at a time.
- Round-robin assignment: doc 0 → val, doc 1 → test, doc 2+ → train
  (until each split's token budget is met; `SplitBudget`).
- Each document is truncated to `MAX_SEQ_LEN` (with EOS appended) or
  padded with `PAD_ID` to `MAX_SEQ_LEN`.
- Packs `SEQUENCES_PER_SHARD = 4096` sequences per shard →
  `data/shards/<split>/shard_*.npy` (shape `(4096, 4096)`, dtype `int32`).
- Writes `manifest.json` (YAML-encoded despite the `.json` name) with
  per-split shard lists, sequence counts, token counts, and per-source
  budgets.

## Stage 6 — `streaming_dataloader.py`

- `load_manifest` reads `manifest.json`; `_open_shards` memory-maps each
  shard and asserts `dtype == int32`, `ndim == 2`, `shape[1] == MAX_SEQ_LEN`.
- `make_dataloader(split, micro_batch_size, seq_len, shuffle, seed,
  drop_last, infinite, shards_dir, device)`:
  - Builds a flat list of `(shard_idx, seq_idx)` pairs.
  - Shuffles the order each epoch with a seeded `random.Random`.
  - Yields `(tokens, targets)` where `tokens` is `(B, T)` `long` and
    `targets = roll(tokens, -1, dims=1)` with the last column set to
    `EOS_ID` (so `targets[t] = tokens[t+1]` for `t < T-1` and
    `targets[T-1] = EOS`).
  - Optional `device` triggers non-blocking H2D transfer.
- `train_dataloader` / `val_dataloader` / `test_dataloader` are thin
  wrappers with sensible defaults (train: shuffle + infinite; val/test:
  no shuffle + finite).

## `prepare_data.py` — the FusionLLM shim

`data/prepare_data.py` is a thin entry point that delegates to a shared
universal pipeline (`shared_data.prepare_data.run_pipeline`). It
materializes a project-local `data/data_config.yaml` with FusionLLM's
tokenizer settings (`fusionllm-bpe-64k`, vocab 64,000, EOS=0, PAD=2),
then calls `run_pipeline` with the universal mixture and the
FusionLLM-specific data config.

The cached **clean JSONL** can be shared across all LLM projects, but the
**tokens** must be regenerated per project because each project's
tokenizer produces different token IDs.