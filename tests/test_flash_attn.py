"""Unit tests for `kernels/flash_attn`.

Phase 5.2:
* ``has_flash_attn`` returns bool (False on CPU).
* ``flash_attention`` fallback to SDPA matches direct SDPA call.
* ``long_short_window_mask`` produces correct masks.
* MLA with ``use_fa3=True`` still runs (fallback to SDPA on CPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.flash_attn import flash_attention, has_flash_attn, long_short_window_mask
from models.mla import MultiHeadLatentAttention


class TestFlashAttnDetection:
    def test_has_flash_attn_returns_bool(self):
        result = has_flash_attn()
        assert isinstance(result, bool)
        if torch.cuda.is_available():
            pass
        else:
            assert result is False


class TestFlashAttentionFallback:
    @pytest.fixture
    def qkv(self):
        torch.manual_seed(42)
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        return q, k, v

    def test_fallback_matches_sdpa(self, qkv):
        q, k, v = qkv
        ref = F.scaled_dot_product_attention(q, k, v)
        out = flash_attention(q, k, v, use_fa3=False)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_fallback_with_scale(self, qkv):
        q, k, v = qkv
        ref = F.scaled_dot_product_attention(q, k, v, scale=0.125)
        out = flash_attention(q, k, v, scale=0.125, use_fa3=False)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_fallback_with_mask(self, qkv):
        q, k, v = qkv
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :, 4:] = float("-inf")
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = flash_attention(q, k, v, attn_mask=mask, use_fa3=False)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_fa3_flag_true_falls_back_on_cpu(self, qkv):
        q, k, v = qkv
        out = flash_attention(q, k, v, use_fa3=True)
        ref = F.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(ref, out, atol=1e-5)


class TestLongShortWindowMask:
    def test_returns_correct_length(self):
        masks = long_short_window_mask(6, 8, 8, "cpu", window=4, period=3)
        assert len(masks) == 6

    def test_global_at_period(self):
        masks = long_short_window_mask(6, 8, 8, "cpu", window=4, period=3)
        assert masks[2] is None
        assert masks[5] is None

    def test_non_global_has_mask(self):
        masks = long_short_window_mask(6, 8, 8, "cpu", window=4, period=3)
        assert masks[0] is not None
        assert masks[1] is not None
        assert masks[3] is not None
        assert masks[4] is not None

    def test_mask_shape(self):
        masks = long_short_window_mask(4, 8, 8, "cpu", window=4, period=2)
        assert masks[0].shape == (1, 1, 8, 8)


class TestMLAFA3Flag:
    def test_fa3_flag_false(self):
        cfg = dict(
            dim=32,
            n_heads=4,
            n_kv_groups=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            max_seq_len=16,
            use_fa3=False,
        )
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        assert mla.use_fa3 is False

    def test_fa3_flag_true(self):
        cfg = dict(
            dim=32,
            n_heads=4,
            n_kv_groups=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            max_seq_len=16,
            use_fa3=True,
        )
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        assert mla.use_fa3 is True

    def test_mla_with_fa3_flag_runs(self):
        cfg = dict(
            dim=32,
            n_heads=4,
            n_kv_groups=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            max_seq_len=16,
            use_fa3=True,
        )
        mla = MultiHeadLatentAttention(cfg, layer_idx=0, world_size=1, rank=0)
        x = torch.randn(1, 4, 32)
        y = mla(x, start_pos=0, use_cache=False)
        assert y.shape == (1, 4, 32)
        assert torch.isfinite(y).all()
