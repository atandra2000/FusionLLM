# data/common.py
"""Shared utilities for the FusionLLM-v1 data pipeline."""

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

VOCAB_SIZE: int = 64_000
MAX_SEQ_LEN: int = 4096
NUM_SPECIAL_TOKENS: int = 32
EOS_ID: int = 0
BOS_ID: int = 1
PAD_ID: int = 2
UNK_ID: int = 3
RESERVED_ID_START: int = 4
RESERVED_ID_END: int = 32

DATA_ROOT: Path = Path("data")
CONFIG_ROOT: Path = DATA_ROOT / "config"
RAW_ROOT: Path = DATA_ROOT / "raw"
CLEAN_ROOT: Path = DATA_ROOT / "clean"
TOKENS_ROOT: Path = DATA_ROOT / "tokens"
SHARDS_ROOT: Path = DATA_ROOT / "shards"
TOKENIZER_DIR: Path = DATA_ROOT / "tokenizer"
STATE_ROOT: Path = DATA_ROOT / "state"
LOGS_ROOT: Path = DATA_ROOT / "logs"

MASTER_SEED: int = 1729
SHARD_DTYPE: Any = np.int32


def seed_everything(seed: int = MASTER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def load_state(stage: str) -> dict:
    path = STATE_ROOT / f"{stage}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(stage: str, state: dict) -> None:
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


_LOG_PATH: Path | None = None


def log(msg: str, *, to_console: bool = True) -> None:
    if to_console:
        print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOGS_ROOT / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def open_text_writer(path: str | Path) -> "zstd.ZstdCompressionWriter":
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "wb")
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.stream_writer(fh, closefd=True)


def open_text_reader(path: str | Path) -> "zstd.ZstdDecompressionReader":
    fh = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    return dctx.stream_reader(fh, closefd=True)


def stable_hash_u64(text: str) -> int:
    s = unicodedata.normalize("NFC", text)
    h = 1469598103934665603
    for ch in s[:8192]:
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def json_escape(s: str) -> str:
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


def iter_source_shards(root: Path, source_id: str) -> Iterator[Path]:
    root = root / source_id
    if not root.exists():
        return
    yield from sorted(root.glob("*.jsonl.zst"))


def iter_clean_shards(source_id: str) -> Iterator[Path]:
    return iter_source_shards(CLEAN_ROOT, source_id)


def iter_clean_jsonl(path: Path) -> Iterator[dict]:
    with open_text_reader(path) as reader:
        data = reader.read()
    for line in data.splitlines():
        line = line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


@dataclass
class SourceBudget:
    source_id: str
    weight: float
    target_train_tokens: int
    target_val_tokens: int
    target_test_tokens: int


def compute_source_budgets(mixture: dict) -> list[SourceBudget]:
    total = int(mixture["total_tokens"])
    train_frac = float(mixture["train_fraction"])
    val_frac = float(mixture["val_fraction"])
    test_frac = float(mixture["test_fraction"])
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    train_total = int(total * train_frac)
    val_total = int(total * val_frac)
    test_total = total - train_total - val_total

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
    for src in mixture["sources"]:
        if src["id"] == source_id:
            return src
    raise KeyError(f"Unknown source id: {source_id}")


def progress(iterable: Iterable, *, desc: str, total: int | None = None) -> Iterator:
    return tqdm(iterable, desc=desc, total=total, unit="doc", smoothing=0.1)


_PII_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PII_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PII_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def light_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_CHARS.sub("", text)
    text = _PII_EMAIL.sub(" ", text)
    text = _PII_IPV4.sub(" ", text)
    text = _PII_PHONE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
