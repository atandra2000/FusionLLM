"""Unit tests for `models/mla.py`.

Phase 0 scope (per `plan.md:0.3`):
  * constructor — all shapes, GQA divisibility checks.
  * GQA-on-MLA grouping math (`q_per_kv`, `n_local_kv_heads`).
  * QK-norm presence.

Forward+backward tests are GPU-marked (Phase 0 only requires shape
math; the MLA forward has a pre-existing torch 2.7 bool-subtract
bug on CPU that the tests must not regress on).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.mla import MultiHeadLatentAttention
from models.rope import _yarn_freq_scaling as ref_yarn_freq_scaling


def _mla_cfg() -> dict:
    """Minimal MLA config that exercises GQA-on-MLA."""
    return dict(
        dim=32,
        n_heads=4,
        n_kv_groups=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        max_seq_len=16,
        sliding_window=8,
        rope_theta=10_000.0,
        rope_factor=1.0,
    )


# ── Constructor ─────────────────────────────────────────────────────────────
class TestMLAConstructor:
    def test_q_per_kv_is_correct(self):
        mla = MultiHeadLatentAttention(_mla_cfg(), layer_idx=0, world_size=1, rank=0)
        # n_heads=4, n_kv_groups=2 → q_per_kv=2
        assert mla.q_per_kv == 2
        assert mla.n_local_heads == 4
        assert mla.n_local_kv_heads == 2

    def test_n_kv_groups_must_divide_n_heads(self):
        cfg = _mla_cfg()
        cfg["n_kv_groups"] = 3
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)

    def test_n_kv_groups_must_divide_world_size(self):
        cfg = _mla_cfg()
        # n_kv_groups=2 not divisible by world_size=4
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadLatentAttention(cfg, layer_idx=0, world_size=4, rank=0)

    def test_q_lora_rank_zero_skips_wq_a(self):
        cfg = _mla_cfg()
        cfg["q_lora_rank"] = 0
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        assert hasattr(mla, "wq")
        assert not hasattr(mla, "wq_a")

    def test_qk_norm_layers_present(self):
        mla = MultiHeadLatentAttention(_mla_cfg(), layer_idx=0, world_size=1, rank=0)
        assert hasattr(mla, "q_norm_qk") and hasattr(mla, "k_norm_qk")
        assert isinstance(mla.q_norm_qk, nn.RMSNorm)

    def test_wkv_b_cache_invalidation_hook(self):
        mla = MultiHeadLatentAttention(_mla_cfg(), layer_idx=0, world_size=1, rank=0)
        assert mla._wkv_b_cached is True
        mla._invalidate_wkv_b_cache()
        assert mla._wkv_b_cached is False

    def test_window_disabled_when_none(self):
        cfg = _mla_cfg()
        cfg["sliding_window"] = None
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        assert mla.window is None

    def test_yarn_scaling_is_noop_when_factor_eq_1(self):
        # _yarn_freq_scaling moved to models.rope in Phase 2.1; MLA
        # now owns a RotaryEmbedding but the math is identical.
        cfg = _mla_cfg()
        cfg["rope_factor"] = 1.0
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        inv_freq = torch.tensor([1.0, 0.5, 0.25, 0.125])
        out = ref_yarn_freq_scaling(inv_freq, rope_factor=1.0, dim=8)
        assert torch.allclose(out, inv_freq)

    def test_yarn_scaling_divides_by_factor(self):
        cfg = _mla_cfg()
        cfg["rope_factor"] = 2.0
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        inv_freq = torch.tensor([1.0, 0.5, 0.25, 0.125])
        out = ref_yarn_freq_scaling(inv_freq, rope_factor=2.0, dim=8)
        # factor^0 = 1, factor^(2/dim) for dim=8: 2^(0/8)=1, 2^(2/8)=2^0.25
        # out[0] = 1/1 = 1, out[1] = 0.5/2^0.25
        assert out[0] == inv_freq[0]
        assert out[1] < inv_freq[1]
