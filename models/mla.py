# models/mla.py
"""Multi-Head Latent Attention.

Architecture:
  Input (B, T, 768)
    ├─ wq_a: Linear(768→192) + RMSNorm(192) + wq_b: Linear(192→12×96=1152)
    │         └─ Split: Q_nope (12, 64), Q_pe (12, 32) → RoPE(Q_pe)
    ├─ wkv_a: Linear(768→128) → Split: KV_latent (96), K_pe (32) → RoPE(K_pe)
    │         └─ RMSNorm(96) → wkv_b: Linear(96→8×128=1024)
    │                    └─ Split: K_nope (8, 64), V (8, 64)
    ├─ Absorption: Q_nope @ wkv_b_k  →  (B, T, 12, kv_lora_rank=96)
    ├─ GQA expand K/V: 8 → 12 groups
    ├─ Concat: Q = [Q_nope_proj, Q_pe], K = [KV_normed, K_pe]
    ├─ QK-Norm: RMSNorm on Q and K (dim = kv_lora_rank + qk_rope_head_dim = 128)
    ├─ SDPA
    └─ wo: Linear(768→768)

Per-layer params: 1,155,616
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cached_len and self._cached_cos is not None and self._cached_cos.device == device:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE. x: (..., seq_len, head_dim)."""
        seq_len = x.size(-2)
        device = x.device
        self._build_cache(start_pos + seq_len, device)
        cos = self._cached_cos[start_pos: start_pos + seq_len].to(x.dtype)
        sin = self._cached_sin[start_pos: start_pos + seq_len].to(x.dtype)
        # Reshape for broadcasting: (1, 1, seq_len, head_dim) for 4D, etc.
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_half = x.float().reshape(*x.shape[:-1], -1, 2)
        x_rot = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).flatten(-2)
        return (x * cos + x_rot * sin).to(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with GQA-on-top-of-MLA."""

    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        d = config["dim"]                           # 768
        n_heads = config["n_heads"]                 # 12
        n_kv_groups = config["n_kv_groups"]         # 8
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups

        self.q_lora_rank = config["q_lora_rank"]            # 192
        self.kv_lora_rank = config["kv_lora_rank"]          # 96
        self.qk_nope_head_dim = config["qk_nope_head_dim"]  # 64
        self.qk_rope_head_dim = config["qk_rope_head_dim"]  # 32
        self.v_head_dim = config["v_head_dim"]              # 64
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 96
        self.max_seq_len = config["max_seq_len"]            # 4096

        # ── Query low-rank projection ────────────────────────────────────
        self.wq_a = nn.Linear(d, self.q_lora_rank, bias=False)       # 768→192
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)         # 192
        self.wq_b = nn.Linear(self.q_lora_rank, n_heads * self.qk_head_dim, bias=False)  # 192→1152

        # ── KV latent compression ────────────────────────────────────────
        self.wkv_a = nn.Linear(d, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)  # 768→128
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)        # 96
        # wkv_b: (kv_lora_rank) → n_kv_groups × (qk_nope_head_dim + v_head_dim) = 8×128=1024
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            n_kv_groups * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # ── Output projection ────────────────────────────────────────────
        self.wo = nn.Linear(n_heads * self.v_head_dim, d, bias=False)  # 768→768

        # ── QK-Norm (applied to Q/K concat: dim = kv_lora_rank + qk_rope_head_dim) ──
        self.qk_norm_dim = self.kv_lora_rank + self.qk_rope_head_dim                     # 96+32=128
        self.q_norm_qk = nn.RMSNorm(self.qk_norm_dim, eps=1e-6)
        self.k_norm_qk = nn.RMSNorm(self.qk_norm_dim, eps=1e-6)

        # ── RoPE ─────────────────────────────────────────────────────────
        self.rope = RotaryEmbedding(
            head_dim=self.qk_rope_head_dim,
            max_seq_len=self.max_seq_len,
            theta=config.get("rope_theta", 10000.0),
        )

        # Precompute KV group → Q head mapping for GQA
        # n_heads=12, n_kv_groups=8. 4 groups get 2 heads, 4 get 1 head.
        base = n_heads // n_kv_groups
        remainder = n_heads % n_kv_groups
        kv_group_for_q = torch.zeros(n_heads, dtype=torch.long)
        idx = 0
        for g in range(n_kv_groups):
            n_q = base + (1 if g < remainder else 0)
            kv_group_for_q[idx: idx + n_q] = g
            idx += n_q
        self.register_buffer("_kv_group_for_q", kv_group_for_q, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        n_heads = self.n_heads
        n_kv_groups = self.n_kv_groups
        kv_lora_rank = self.kv_lora_rank

        # ── 1. Query ─────────────────────────────────────────────────────
        q = self.wq_b(self.q_norm(self.wq_a(x)))           # (B, T, n_heads * qk_head_dim)
        q = q.view(B, T, n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope(q_pe, start_pos)                 # (B, T, n_heads, qk_rope_head_dim)

        # ── 2. KV latent compression ─────────────────────────────────────
        kv_a = self.wkv_a(x)                             # (B, T, kv_lora_rank + qk_rope_head_dim)
        kv_latent, k_pe_raw = kv_a.split([kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed = self.kv_norm(kv_latent)             # (B, T, kv_lora_rank=96)
        k_pe = self.rope(k_pe_raw.unsqueeze(-2), start_pos).squeeze(-2)  # (B, T, qk_rope_head_dim)

        # ── 3. Absorption trick ──────────────────────────────────────────
        # wkv_b weight: (n_kv_groups * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        # Reshape to (n_kv_groups, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        wkv_b_w = self.wkv_b.weight.view(
            n_kv_groups,
            self.qk_nope_head_dim + self.v_head_dim,
            kv_lora_rank,
        )
        # Split into K and V parts
        wkv_b_k = wkv_b_w[:, :self.qk_nope_head_dim, :]   # (8, 64, 96)  — produces K_nope
        wkv_b_v = wkv_b_w[:, self.qk_nope_head_dim:, :]   # (8, 64, 96)  — produces V

        # For absorption: project Q_nope through wkv_b_k into latent space
        # Each Q head uses its KV group's wkv_b_k slice
        group_idx = self._kv_group_for_q.to(device)  # (n_heads,)
        wkv_b_k_q = wkv_b_k[group_idx]   # (n_heads, 64, kv_lora_rank)
        wkv_b_v_q = wkv_b_v[group_idx]   # (n_heads, 64, kv_lora_rank)

        # Absorb: Q_nope_proj = Q_nope @ wkv_b_k  (in latent space)
        # q_nope: (B, T, 12, 64), wkv_b_k_q: (12, 64, 96)
        # → q_nope_proj: (B, T, 12, 96) = (B, T, n_heads, kv_lora_rank)
        q_nope_proj = torch.einsum("bthd,hdc->bthc", q_nope, wkv_b_k_q)

        # V from latent: kv_normed (B,T,96) @ wkv_b_v_q (n_heads,96,64)^T
        # → (B, T, n_heads, v_head_dim)
        v = torch.einsum("btc,hdc->bthd", kv_normed, wkv_b_v_q)

        # ── 4. Prepare Q, K, V for SDPA ──────────────────────────────────
        # Q = concat(q_nope_proj, q_pe) → (B, T, 12, 96+32=128)
        q_concat = torch.cat([q_nope_proj, q_pe], dim=-1)

        # K = concat(kv_normed, k_pe) expanded to n_heads
        # kv_normed: (B,T,96) → (B,T,1,96) → (B,T,12,96)
        kv_expanded = kv_normed.unsqueeze(2).expand(-1, -1, n_heads, -1)
        # k_pe: (B,T,32) → (B,T,1,32) → (B,T,12,32)
        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, n_heads, -1)
        k_concat = torch.cat([kv_expanded, k_pe_expanded], dim=-1)  # (B,T,12,128)

        # ── 5. QK-Norm ───────────────────────────────────────────────────
        q_concat = self.q_norm_qk(q_concat)
        k_concat = self.k_norm_qk(k_concat)

        # ── 6. SDPA ──────────────────────────────────────────────────────
        q_sdpa = q_concat.transpose(1, 2)  # (B, 12, T, 128)
        k_sdpa = k_concat.transpose(1, 2)  # (B, 12, T, 128)
        v_sdpa = v.transpose(1, 2)         # (B, 12, T, 64)

        attn_out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=mask,
            is_causal=(mask is None),
            dropout_p=0.0,
        )  # (B, 12, T, 64)

        # ── 7. Output ────────────────────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, 768)
        return self.wo(attn_out)
