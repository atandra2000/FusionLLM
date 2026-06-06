"""Shared pytest fixtures and configuration for the test suite.

Goals
-----
1. CPU-only by default. GPU tests must opt in with `@pytest.mark.gpu`.
2. Deterministic seeds where possible.
3. A small, fast default model config (`tiny_cfg`) that exercises every
   architectural unit without being slow.

See `plan.md:0.3` for the test layout.
"""

from __future__ import annotations

import os
import random

import pytest
import torch


# ── Determinism ─────────────────────────────────────────────────────────────
def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def _seed_every_test():
    """Re-seed before every test for reproducibility."""
    _set_seed(0)
    torch.use_deterministic_algorithms(False)
    yield


# ── Tiny model config (CPU-friendly) ───────────────────────────────────────
@pytest.fixture
def tiny_cfg() -> dict:
    """A minimal config that exercises MLA, MoE and Mamba.

    Kept deliberately small so forward+backward takes < 1 s on CPU.
    """
    return dict(
        vocab_size=128,
        dim=32,
        n_layers=6,
        max_seq_len=16,
        layer_schedule="5:1",  # 5 MLA + 1 Mamba per period
        # MLA
        n_heads=4,
        n_kv_groups=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        rope_theta=10_000.0,
        rope_factor=1.0,
        qk_norm=True,
        sliding_window=8,
        # Dense FFN (Mamba layer)
        inter_dim=64,
        ffn_activation="swiglu",
        # MoE
        n_routed_experts=4,
        n_shared_experts=2,
        n_activated_experts=2,
        moe_inter_dim=32,
        expert_capacity_factor=1.5,
        expert_dropout_prob=0.0,
        moe_warmup_steps=0,
        n_expert_groups=2,
        n_limited_groups=1,
        group_topk=1,
        route_scale=1.0,
        bias_upper_threshold=0.10,
        bias_lower_threshold=0.10,
        moe_activation="swiglu",
        # Mamba
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_headdim=8,
        # MTP
        mtp_depth=0,
        mtp_loss_weight=0.0,
        # Embed / head
        tie_embeddings=True,
        no_bias_linear=True,
    )


# ── GPU guard ──────────────────────────────────────────────────────────────
@pytest.fixture
def requires_gpu():
    """Skip the test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — marked @pytest.mark.gpu")
    return torch.device("cuda:0")
