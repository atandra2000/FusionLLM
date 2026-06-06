#!/usr/bin/env python3
"""Benchmark MLA BF16 throughput.

Usage
-----
    python scripts/bench_mla.py                          # BF16 only (CPU)
    python scripts/bench_mla.py --n-steps 200             # custom steps
    python scripts/bench_mla.py --profile a100            # full GPU bench

On macOS / CPU this script measures the fallback path only.  On an
A100 system, this benchmarks BF16 throughput.

Output (stdout)
---------------
    MLA bench (bsz=2, seqlen=4096, n_heads=32, head_dim=256)
    BF16 : 1234.5 tok/s
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

BENCH_PROFILES: dict[str, dict[str, Any]] = {
    "a100": {
        "dim": 2048,
        "n_heads": 32,
        "n_kv_groups": 8,
        "q_lora_rank": 512,
        "kv_lora_rank": 256,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "max_seq_len": 8192,
        "batch_size": 2,
        "n_steps": 100,
    },
    "micro": {
        "dim": 256,
        "n_heads": 4,
        "n_kv_groups": 2,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "max_seq_len": 512,
        "batch_size": 1,
        "n_steps": 50,
    },
}


def bench_mla(
    profile: str | None = None,
    n_steps: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    if profile is not None:
        cfg = BENCH_PROFILES[profile]
    else:
        cfg = BENCH_PROFILES["micro"]
    if n_steps is not None:
        cfg["n_steps"] = n_steps

    bsz = cfg["batch_size"]
    seqlen = cfg["max_seq_len"]
    dim = cfg["dim"]
    model_cfg = {k: v for k, v in cfg.items() if k not in ("batch_size", "n_steps")}

    torch.manual_seed(0)
    dev = torch.device(device)

    from models.mla import MultiHeadLatentAttention

    mla = MultiHeadLatentAttention(model_cfg, layer_idx=0, world_size=1, rank=0).to(dev)
    mla.eval()

    x = torch.randn(bsz, seqlen, dim, device=dev)

    # Warmup
    for _ in range(5):
        _ = mla(x, start_pos=0, use_cache=False)

    torch.cuda.synchronize() if device == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(cfg["n_steps"]):
        _ = mla(x, start_pos=0, use_cache=False)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - start

    tokens = bsz * seqlen * cfg["n_steps"]
    tok_s = tokens / elapsed

    print(f"  BF16 : {tok_s:.1f} tok/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLA throughput benchmark")
    parser.add_argument(
        "--profile", type=str, default=None, choices=list(BENCH_PROFILES),
        help="Benchmark profile (default: micro for CPU)"
    )
    parser.add_argument("--n-steps", type=int, default=None, help="Override step count")
    args = parser.parse_args()

    print(
        "MLA bench "
        f"(profile={args.profile or 'micro'}, "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'})"
    )
    bench_mla(profile=args.profile, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
