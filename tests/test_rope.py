"""Unit tests for `models/rope.py`.

Validates the extracted RoPE core against a frozen reference
implementation (the original `mla._apply_rope` / `_yarn_freq_scaling`
logic).  All tests run on CPU since RoPE is dtype-only.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.rope import RotaryEmbedding, _yarn_freq_scaling, apply_rope


# ── Reference implementation (frozen) ──────────────────────────────────────
def ref_yarn_freq_scaling(inv_freq: torch.Tensor, rope_factor: float, dim: int) -> torch.Tensor:
    if rope_factor <= 1.0:
        return inv_freq.clone()
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=inv_freq.device)
    scale = rope_factor ** (i / dim)
    return inv_freq / scale


def ref_extend_rope(head_dim: int, end_pos: int, rope_theta: float, rope_factor: float, device):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    inv_freq = ref_yarn_freq_scaling(inv_freq, rope_factor, head_dim)
    t = torch.arange(end_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def ref_apply_rope(
    x: torch.Tensor, start_pos: int, seqlen: int, freqs_cis: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[start_pos : start_pos + seqlen].view(1, seqlen, 1, -1)
    return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)


# ── YaRN frequency scaling ─────────────────────────────────────────────────
class TestYarnScaling:
    def test_noop_when_factor_eq_1(self):
        inv = torch.tensor([1.0, 0.5, 0.25])
        out = _yarn_freq_scaling(inv, rope_factor=1.0, dim=8)
        assert torch.allclose(out, inv)

    def test_factor_gt_1_compresses_high_freq(self):
        inv = torch.tensor([1.0, 0.5, 0.25, 0.125])
        out = _yarn_freq_scaling(inv, rope_factor=4.0, dim=8)
        # factor^0 = 1 → out[0] = inv[0]; out[1] = inv[1] / 4^0.25 < inv[1]
        assert out[0] == inv[0]
        assert out[1] < inv[1]
        assert out[2] < inv[2]
        assert out[3] < inv[3]

    def test_matches_reference(self):
        inv = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        ref = ref_yarn_freq_scaling(inv, rope_factor=2.0, dim=12)
        out = _yarn_freq_scaling(inv, rope_factor=2.0, dim=12)
        assert torch.allclose(out, ref, atol=1e-7)


# ── apply_rope ─────────────────────────────────────────────────────────────
class TestApplyRope:
    def test_no_rotation_when_freq_is_zero(self):
        # When all freqs are 0, exp(0i) = 1+0i → identity rotation.
        bsz, seqlen, n_heads, head_dim = 1, 4, 2, 4
        x = torch.randn(bsz, seqlen, n_heads, head_dim)
        freqs = torch.polar(torch.ones(8, head_dim // 2), torch.zeros(8, head_dim // 2))
        y = apply_rope(x, 0, seqlen, freqs)
        # Output should match input (allowing cast jitter).
        assert torch.allclose(y, x, atol=1e-5)

    def test_matches_reference(self):
        torch.manual_seed(0)
        bsz, seqlen, n_heads, head_dim = 2, 8, 3, 16
        x = torch.randn(bsz, seqlen, n_heads, head_dim)
        head_dim_rope = 8
        # We must use a freqs table of size matching head_dim_rope for the ref.
        freqs = ref_extend_rope(
            head_dim_rope, end_pos=32, rope_theta=10_000.0, rope_factor=1.0, device=x.device
        )
        # apply_rope uses head_dim = freqs.shape[-1] * 2.
        # We construct x with the matching head_dim.
        x = torch.randn(bsz, seqlen, n_heads, head_dim_rope)
        ref = ref_apply_rope(x, 0, seqlen, freqs)
        out = apply_rope(x, 0, seqlen, freqs)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_start_pos_offsets_correctly(self):
        torch.manual_seed(0)
        bsz, seqlen, n_heads, head_dim = 1, 4, 1, 8
        x = torch.randn(bsz, seqlen, n_heads, head_dim)
        freqs = ref_extend_rope(
            head_dim, end_pos=8, rope_theta=10_000.0, rope_factor=1.0, device=x.device
        )
        y0 = apply_rope(x, 0, seqlen, freqs)
        y2 = apply_rope(x, 2, seqlen, freqs)
        assert not torch.allclose(y0, y2, atol=1e-3)

    def test_dtype_preserved(self):
        x = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
        freqs = ref_extend_rope(8, end_pos=4, rope_theta=10_000.0, rope_factor=1.0, device=x.device)
        y = apply_rope(x, 0, 4, freqs)
        assert y.dtype == torch.bfloat16


# ── RotaryEmbedding module ─────────────────────────────────────────────────
class TestRotaryEmbedding:
    def test_constructs_with_default_args(self):
        rope = RotaryEmbedding(head_dim=8, rope_theta=10_000.0, max_seq_len=128)
        assert rope.freqs_cis.numel() == 0  # not built until first use

    def test_extend_to_grows_table(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64)
        rope.extend_to(16, torch.device("cpu"))
        assert rope.freqs_cis.shape == (64, 4)  # grows geometrically
        rope.extend_to(40, torch.device("cpu"))
        assert rope.freqs_cis.shape[0] >= 40
        assert rope.freqs_cis.shape[0] <= 64

    def test_extend_to_caps_at_max_seq_len(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
        with pytest.raises(RuntimeError, match="end_pos.*max_seq_len"):
            rope.extend_to(32, torch.device("cpu"))

    def test_extend_to_is_idempotent_when_already_covered(self):
        rope = RotaryEmbedding(head_dim=8, max_seq_len=64)
        rope.extend_to(16, torch.device("cpu"))
        f1 = rope.freqs_cis.clone()
        rope.extend_to(8, torch.device("cpu"))  # smaller, already covered
        assert torch.equal(rope.freqs_cis, f1)

    def test_forward_matches_apply_rope(self):
        rope = RotaryEmbedding(head_dim=8, rope_theta=10_000.0, max_seq_len=64)
        x = torch.randn(1, 4, 2, 8)
        y_module = rope(x, 0, 4)
        # Build the same freqs_cis and call apply_rope directly
        freqs = ref_extend_rope(
            8, end_pos=64, rope_theta=10_000.0, rope_factor=1.0, device=x.device
        )
        y_direct = apply_rope(x, 0, 4, freqs)
        assert torch.allclose(y_module, y_direct, atol=1e-5)

    def test_yarn_factor_changes_freqs(self):
        r1 = RotaryEmbedding(head_dim=8, rope_factor=1.0, max_seq_len=32)
        r2 = RotaryEmbedding(head_dim=8, rope_factor=2.0, max_seq_len=32)
        r1.extend_to(16, torch.device("cpu"))
        r2.extend_to(16, torch.device("cpu"))
        assert not torch.allclose(r1.freqs_cis, r2.freqs_cis)

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError, match="head_dim must be even"):
            RotaryEmbedding(head_dim=7, max_seq_len=32)
