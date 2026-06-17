# models/mla.py
"""Multi-Head Latent Attention (Flash Attention 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cached_len and self._cached_cos is not None:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE."""
        seq_len = x.size(-2)
        self._build_cache(start_pos + seq_len, x.device)
        cos = self._cached_cos[start_pos: start_pos + seq_len].to(x.dtype)
        sin = self._cached_sin[start_pos: start_pos + seq_len].to(x.dtype)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_half = x.float().reshape(*x.shape[:-1], -1, 2)
        x_rot = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).flatten(-2)
        return (x * cos + x_rot * sin).to(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with GQA."""

    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        d, n_heads, n_kv_groups = config["dim"], config["n_heads"], config["n_kv_groups"]
        self.n_heads, self.n_kv_groups = n_heads, n_kv_groups
        self.q_lora_rank = config["q_lora_rank"]
        self.kv_lora_rank = config["kv_lora_rank"]
        self.qk_nope_head_dim = config["qk_nope_head_dim"]
        self.qk_rope_head_dim = config["qk_rope_head_dim"]
        self.v_head_dim = config["v_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_seq_len = config["max_seq_len"]

        self.wq_a = nn.Linear(d, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
        self.wq_b = nn.Linear(self.q_lora_rank, n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(d, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.wkv_b = nn.Linear(self.kv_lora_rank, n_kv_groups * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(n_heads * self.v_head_dim, d, bias=False)

        self.qk_norm_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.q_norm_qk = nn.RMSNorm(self.qk_norm_dim, eps=1e-6)
        self.k_norm_qk = nn.RMSNorm(self.qk_norm_dim, eps=1e-6)
        self.rope = RotaryEmbedding(head_dim=self.qk_rope_head_dim, max_seq_len=self.max_seq_len,
        theta=config.get("rope_theta", 10000.0))

        base, remainder = n_heads // n_kv_groups, n_heads % n_kv_groups
        kv_group_for_q = torch.zeros(n_heads, dtype=torch.long)
        idx = 0
        for g in range(n_kv_groups):
            n_q = base + (1 if g < remainder else 0)
            kv_group_for_q[idx: idx + n_q] = g
            idx += n_q
        self.register_buffer("_kv_group_for_q", kv_group_for_q, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        n_heads, n_kv_groups, kv_lora_rank = self.n_heads, self.n_kv_groups, self.kv_lora_rank

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope(q_pe, start_pos)

        kv_a = self.wkv_a(x)
        kv_latent, k_pe_raw = kv_a.split([kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed = self.kv_norm(kv_latent)
        k_pe = self.rope(k_pe_raw.unsqueeze(-2), start_pos).squeeze(-2)

        wkv_b_w = self.wkv_b.weight.view(n_kv_groups, self.qk_nope_head_dim + self.v_head_dim, kv_lora_rank)
        wkv_b_k, wkv_b_v = wkv_b_w[:, :self.qk_nope_head_dim, :], wkv_b_w[:, self.qk_nope_head_dim:, :]
        group_idx = self._kv_group_for_q.to(x.device)
        wkv_b_k_q, wkv_b_v_q = wkv_b_k[group_idx], wkv_b_v[group_idx]

        q_nope_proj = torch.einsum("bthd,hdc->bthc", q_nope, wkv_b_k_q)
        v = torch.einsum("btc,hdc->bthd", kv_normed, wkv_b_v_q)

        q_concat = torch.cat([q_nope_proj, q_pe], dim=-1)
        kv_expanded = kv_normed.unsqueeze(2).expand(-1, -1, n_heads, -1)
        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, n_heads, -1)
        k_concat = torch.cat([kv_expanded, k_pe_expanded], dim=-1)

        q_concat = self.q_norm_qk(q_concat)
        k_concat = self.k_norm_qk(k_concat)

        # Flash Attention 2 (with SDPA fallback)
        if flash_attn_func is not None:
            q_fa = q_concat.transpose(1, 2)  # (B, T, 12, 128)
            k_fa = k_concat.transpose(1, 2)  # (B, T, 12, 128)
            v_fa = v.transpose(1, 2)         # (B, T, 12, 64)
            attn_out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            q_fa = q_concat.transpose(1, 2)  # (B, n_heads, T, qk_head_dim)
            k_fa = k_concat.transpose(1, 2)  # (B, n_heads, T, qk_head_dim)
            v_fa = v.transpose(1, 2)         # (B, n_heads, T, v_head_dim)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn_out)
