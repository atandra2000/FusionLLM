# data/common.py
# Shared utilities for the FusionLLM-v1 data pipeline.
#
# Pure Python / stdlib + a few well-known deps (numpy, yaml, tqdm).
# No model, optimizer, or trainer imports.

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml
import zstandard as zstd
from tqdm import tqdm

# ── Frozen constants from the architecture spec ─────────────────────────
VOCAB_SIZE: int = 64_000
MAX_SEQ_LEN: int = 4096
NUM_SPECIAL_TOKENS: int = 32
EOS_ID: int = 0          # <|endoftext|>
BOS_ID: int = 1          # <|bos|>
PAD_ID: int = 2          # <|pad|>
UNK_ID: int = 3          # <|unk|>
RESERVED_ID_START: int = 4
RESERVED_ID_END: int = 32  # inclusive

# Pipeline paths (relative to repo root)
DATA_ROOT: Path = Path("data")
CONFIG_ROOT: Path = DATA_ROOT / "config"
RAW_ROOT: Path = DATA_ROOT / "raw"
CLEAN_ROOT: Path = DATA_ROOT / "clean"
TOKENS_ROOT: Path = DATA_ROOT / "tokens"
SHARDS_ROOT: Path = DATA_ROOT / "shards"
TOKENIZER_DIR: Path = DATA_ROOT / "tokenizer"
STATE_ROOT: Path = DATA_ROOT / "state"
LOGS_ROOT: Path = DATA_ROOT / "logs"

# Master seed for everything that needs one
MASTER_SEED: int = 1729

# Dataloader dtype on disk
# int32 = best of both worlds: smaller than int64, mmap-friendly,
# directly castable to torch.long at load time. (See SIMPLIFIED_DATA_PIPELINE.md §4.)
SHARD_DTYPE: Any = np.int32


# ── Seeding ──────────────────────────────────────────────────────────────
def seed_everything(seed: int = MASTER_SEED) -> None:
    """Seed all relevant RNGs. Idempotent and safe to call multiple times."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch is seeded by callers that need it (training-side) to keep
    # this module torch-free at import time.


# ── YAML / config loading ────────────────────────────────────────────────
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


# ── Resume state (per-stage JSON) ────────────────────────────────────────
def load_state(stage: str) -> dict:
    """Read `data/state/<stage>.json` if present, else return an empty dict."""
    path = STATE_ROOT / f"{stage}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(stage: str, state: dict) -> None:
    """Atomically write `data/state/<stage>.json`."""
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATE_ROOT / f"{stage}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def clear_state(stage: str) -> None:
    path = STATE_ROOT / f"{stage}.json"
    if path.exists():
        path.unlink()


# ── Logging ──────────────────────────────────────────────────────────────
_LOG_PATH: Path | None = None


def log(msg: str, *, to_console: bool = True) -> None:
    """Print a line and (optionally) append to `data/logs/pipeline.log`."""
    if to_console:
        print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOGS_ROOT / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ── Zstandard helpers (clean text shards) ───────────────────────────────
def open_text_writer(path: str | Path) -> "zstd.ZstdCompressionWriter":  # type: ignore[name-defined]
    """Open a path for compressed-text writing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "wb")
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.stream_writer(fh, closefd=True)


def open_text_reader(path: str | Path) -> "zstd.ZstdDecompressionReader":  # type: ignore[name-defined]
    """Open a path for compressed-text reading.

    The returned object is *not* a line iterator; call `.read()` to get
    the full decompressed bytes (zstd stream_reader does not implement
    `.readline()`). Use `iter_clean_jsonl` for line-by-line access.
    """
    fh = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    return dctx.stream_reader(fh, closefd=True)


# ── Hashing for split assignment ─────────────────────────────────────────
def stable_hash_u64(text: str) -> int:
    """Deterministic 64-bit hash for split assignment.

    We use Python's built-in hash of a salted, normalised string, then
    fall back to a CRC-style mixer to spread bits. This is *not* a
    cryptographic hash, just a stable bucket assignment. Same input
    always gives the same output across processes and platforms
    (because the input string is normalised first).
    """
    s = unicodedata.normalize("NFC", text)
    # xxhash would be faster but adds a dep; we use a tiny inline mixer
    # to keep this module dependency-free at the top.
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    for ch in s[:8192]:  # cap length for speed
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ── JSON escaping for bulk writes ────────────────────────────────────────
def json_escape(s: str) -> str:
    """Minimal JSON string escape. Faster than json.dumps for bulk writes."""
    out = []
    for ch in s:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


# ── Iterators over source shards ─────────────────────────────────────────
def iter_source_shards(root: Path, source_id: str) -> Iterator[Path]:
    """Yield `.jsonl.zst` files for a source under *root* in deterministic order."""
    root = root / source_id
    if not root.exists():
        return
    yield from sorted(root.glob("*.jsonl.zst"))


def iter_clean_shards(source_id: str) -> Iterator[Path]:
    """Yield clean `.jsonl.zst` files for a given source."""
    return iter_source_shards(CLEAN_ROOT, source_id)


def iter_clean_jsonl(path: Path) -> Iterator[dict]:
    """Yield decoded JSON objects from a zstd-compressed JSONL file.

    zstd's stream_reader does not implement `.readline()`, so we read
    the full decompressed buffer and split on newlines. For our 256 MB
    rotation size, this is a few hundred MB of RAM peak per file.
    """
    with open_text_reader(path) as reader:
        data = reader.read()
    for line in data.splitlines():
        line = line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue  # skip malformed lines silently


# ── Token-budget computation ────────────────────────────────────────────
@dataclass
class SourceBudget:
    source_id: str
    weight: float
    target_train_tokens: int
    target_val_tokens: int
    target_test_tokens: int


def compute_source_budgets(mixture: dict) -> list[SourceBudget]:
    """Convert mixture weights into per-source token budgets for each split.

    The mixture's `total_tokens` is the *full* target including val+test
    (8.31B). The split fractions then carve it up.
    """
    total = int(mixture["total_tokens"])
    train_frac = float(mixture["train_fraction"])
    val_frac = float(mixture["val_fraction"])
    test_frac = float(mixture["test_fraction"])
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"Split fractions must sum to 1, got {train_frac + val_frac + test_frac}"
    )

    train_total = int(total * train_frac)
    val_total = int(total * val_frac)
    test_total = total - train_total - val_total  # exact

    budgets: list[SourceBudget] = []
    for src in mixture["sources"]:
        w = float(src["weight"])
        budgets.append(
            SourceBudget(
                source_id=src["id"],
                weight=w,
                target_train_tokens=int(round(train_total * w)),
                target_val_tokens=int(round(val_total * w)),
                target_test_tokens=int(round(test_total * w)),
            )
        )
    return budgets


def source_to_spec(mixture: dict, source_id: str) -> dict:
    """Return the mixture entry for a given source id."""
    for src in mixture["sources"]:
        if src["id"] == source_id:
            return src
    raise KeyError(f"Unknown source id: {source_id}")


# ── Progress helpers ────────────────────────────────────────────────────
def progress(iterable: Iterable, *, desc: str, total: int | None = None) -> Iterator:
    return tqdm(iterable, desc=desc, total=total, unit="doc", smoothing=0.1)


# ── Regex / string cleanup (lightweight, used by preprocess) ────────────
# Conservative PII strip: emails, IPv4, US phone numbers.
_PII_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PII_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PII_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")
# Common mojibake / control characters
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def light_clean(text: str) -> str:
    """Apply NFKC, strip control chars and PII. Conservative: no language ID,
    no min-length filter (those are caller decisions)."""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_CHARS.sub("", text)
    text = _PII_EMAIL.sub(" ", text)
    text = _PII_IPV4.sub(" ", text)
    text = _PII_PHONE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
