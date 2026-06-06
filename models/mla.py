# models/mla.py
"""Multi-Head Latent Attention with GQA on top of the low-rank KV.

Three changes from standard MLA:

* **GQA on top of MLA** (Llama 3, Qwen 2.5, Phi-4-mini pattern).  ``n_kv_groups``
  query heads share a single K/V head; the cached latent is replicated
  across the group.  The absorption trick still applies — ``wkv_b`` is
  folded into ``q_nope`` so attention scores are computed against the
  cached latent directly.  This gives the cache benefit of MLA (~10×
  vs MHA) *and* the per-group sharing of GQA (additional 2–8×).
* **Sliding window option** (``window`` config).  When set, only the
  last ``window`` tokens contribute to attention.  Composes with global
  attention in the layer schedule (Gemma 2 5:1 pattern).
* **QK-norm always on** for training stability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.flash_attn import flash_attention

from .rope import RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        config: dict,
        layer_idx: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.world_size = world_size
        self.rank = rank

        # ── Dimensions ────────────────────────────────────────────────────
        self.dim = config["dim"]
        self.n_heads = config["n_heads"]
        self.q_lora_rank = config["q_lora_rank"]
        self.kv_lora_rank = config["kv_lora_rank"]
        self.qk_nope_head_dim = config["qk_nope_head_dim"]
        self.qk_rope_head_dim = config["qk_rope_head_dim"]
        self.v_head_dim = config["v_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_seq_len = config["max_seq_len"]

        # GQA-on-top-of-MLA: n_kv_groups query heads share one K/V head.
        # n_kv_groups == n_heads   → MHA (original DeepSeek-V3)
        # n_kv_groups == 1         → MQA (one KV head)
        # n_kv_groups in-between  → GQA (Llama 3 / Qwen 2.5 / Phi-4-mini)
        self.n_kv_groups = config.get("n_kv_groups", self.n_heads)
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_groups ({self.n_kv_groups})"
            )
        if self.n_kv_groups % world_size != 0:
            raise ValueError(
                f"n_kv_groups ({self.n_kv_groups}) must be divisible by world_size ({world_size})"
            )
        self.q_per_kv = self.n_heads // self.n_kv_groups
        self.n_local_heads = self.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_groups // world_size

        # Sliding window: None or 0 = global, else local attention over
        # the last `window` tokens (Gemma 2 style).
        self.window = config.get("sliding_window", None)

        # ── RoPE & YaRN config ───────────────────────────────────────────
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.rope_factor = config.get("rope_factor", 1.0)

        mscale_raw = config.get("mscale", 1.0)
        self.mscale = (
            0.1 * mscale_raw * math.log(self.rope_factor) + 1.0
            if self.rope_factor > 1.0
            else mscale_raw
        )

        # ── Softmax scale ─────────────────────────────────────────────────
        self.softmax_scale = self.qk_head_dim**-0.5
        if self.max_seq_len > 4096 and self.mscale != 1.0:
            self.softmax_scale *= self.mscale**2

        # ── Query projections (latent) ────────────────────────────────────
        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.wq_b = nn.Linear(
                self.q_lora_rank,
                self.n_local_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.wq = nn.Linear(self.dim, self.n_local_heads * self.qk_head_dim, bias=False)

        # ── KV projections with latent compression + GQA ──────────────────
        # wkv_a always projects to (kv_lora_rank + qk_rope_head_dim); the
        # latent is per-KV-group.  The wkv_b output is per-KV-group, then
        # the q_per_kv factor is captured by the attention itself (q_nope
        # has shape (n_local_heads, qk_nope_head_dim) and broadcasts).
        self.wkv_a = nn.Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_local_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_local_heads * self.v_head_dim, self.dim, bias=False)

        # ── Cached wkv_b slices (non-persistent; rebuilt on demand) ─────
        self.register_buffer(
            "_wkv_b_k",
            torch.empty(self.n_local_kv_heads, self.qk_nope_head_dim, self.kv_lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "_wkv_b_v",
            torch.empty(self.n_local_kv_heads, self.v_head_dim, self.kv_lora_rank),
            persistent=False,
        )
        self._wkv_b_cached: bool = False
        self._rebuild_wkv_b_cache()
        self.wkv_b.weight.register_post_accumulate_grad_hook(
            lambda _p: self._invalidate_wkv_b_cache()
        )

        # ── KV cache (for generation; training bypasses) ─────────────────
        self._cache_batch: int = 0
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

        # ── RoPE frequency table (shared with GDN etc.) ──────────────────
        self.rope = RotaryEmbedding(
            head_dim=self.qk_rope_head_dim,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            max_seq_len=self.max_seq_len,
        )

        # ── QK-norm (always on) ──────────────────────────────────────────
        self.total_attn_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.q_norm_qk = nn.RMSNorm(self.total_attn_dim, eps=1e-6)
        self.k_norm_qk = nn.RMSNorm(self.total_attn_dim, eps=1e-6)

    # ──────────────────────────────────────────────────────────────────────
    # wkv_b cache helpers
    # ──────────────────────────────────────────────────────────────────────

    def _rebuild_wkv_b_cache(self) -> None:
        w = self.wkv_b.weight.view(
            self.n_local_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        self._wkv_b_k.copy_(w[:, : self.qk_nope_head_dim])  # type: ignore[operator]
        self._wkv_b_v.copy_(w[:, self.qk_nope_head_dim :])  # type: ignore[operator]
        self._wkv_b_cached = True

    def _invalidate_wkv_b_cache(self) -> None:
        self._wkv_b_cached = False

    def _get_wkv_b(self):
        if not self._wkv_b_cached:
            self._rebuild_wkv_b_cache()
        return self._wkv_b_k, self._wkv_b_v

    # ──────────────────────────────────────────────────────────────────────
    # RoPE helpers are in models.rope; MLA owns a `self.rope` instance
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Cache management (generation only)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        need_alloc = (
            self.kv_cache is None
            or bsz > self._cache_batch
            or self.kv_cache.device != device
            or self.kv_cache.dtype != dtype
        )
        if not need_alloc:
            return
        new_bsz = max(bsz, self._cache_batch * 2, 16)
        self.kv_cache = torch.zeros(
            new_bsz,
            self.max_seq_len,
            self.kv_lora_rank,
            device=device,
            dtype=dtype,
        )
        self.pe_cache = torch.zeros(
            new_bsz,
            self.max_seq_len,
            self.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        )
        self._cache_batch = new_bsz

    def reset_cache(self) -> None:
        del self.kv_cache
        del self.pe_cache
        self.kv_cache = None
        self.pe_cache = None
        self._cache_batch = 0

    def prefill_cache(
        self,
        kv_latent: torch.Tensor,
        k_pe: torch.Tensor,
        start_pos: int,
    ) -> None:
        bsz, seqlen, _ = kv_latent.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(f"prefill_cache: end_pos {end_pos} > max_seq_len {self.max_seq_len}")
        self.rope.extend_to(end_pos, kv_latent.device)
        self._ensure_cache(bsz, kv_latent.device, kv_latent.dtype)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_latent
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"Layer {self.layer_idx}: end_pos {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )
        self.rope.extend_to(end_pos, x.device)
        if use_cache:
            self._ensure_cache(bsz, x.device, x.dtype)

        # ── Queries ────────────────────────────────────────────────────────
        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        # (bsz, seqlen, n_local_heads, qk_head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope(q_pe, start_pos, seqlen)

        # ── KV latent compression ──────────────────────────────────────────
        kv_a = self.wkv_a(x)
        kv_latent, k_pe_raw = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed = self.kv_norm(kv_latent)
        k_pe = self.rope(k_pe_raw.unsqueeze(2), start_pos, seqlen).squeeze(
            2
        )  # (bsz, seqlen, qk_rope_head_dim)

        if use_cache:
            self.kv_cache[:bsz, start_pos:end_pos] = kv_normed
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe
            ctx_kv = self.kv_cache[:bsz, :end_pos]
            ctx_pe = self.pe_cache[:bsz, :end_pos]
        else:
            ctx_kv = kv_normed
            ctx_pe = k_pe

        # ── Cached wkv_b slices (absorption trick) ─────────────────────────
        wkv_b_k, wkv_b_v = self._get_wkv_b()  # (n_local_kv_heads, qk_nope/v_head_dim, kv_lora_rank)

        # Project q_nope into the latent space.  GQA: replicate wkv_b_k
        # across the q_per_kv group so that each Q head has its own
        # absorbed-projection matrix, but the K/V head count is small.
        # Shape: (n_local_heads, qk_nope_head_dim, kv_lora_rank)
        if self.q_per_kv > 1:
            wkv_b_k_full = wkv_b_k.repeat_interleave(self.q_per_kv, dim=0)
            wkv_b_v_full = wkv_b_v.repeat_interleave(self.q_per_kv, dim=0)
        else:
            wkv_b_k_full = wkv_b_k
            wkv_b_v_full = wkv_b_v

        q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_k_full)

        # Expand context to per-head shape, replicating K/V across groups.
        # ctx_kv/ctx_pe are (bsz, seqlen, dim) — unsqueeze(2) gives 4-d.
        ctx_kv_exp = ctx_kv.unsqueeze(2).expand(-1, -1, self.n_local_heads, -1)
        ctx_pe_exp = ctx_pe.unsqueeze(2).expand(-1, -1, self.n_local_heads, -1)
        # q_pe is (bsz, seqlen, n_local_heads, head_dim) — no expand needed.

        Q = torch.cat([q_nope_proj, q_pe], dim=-1)
        K = torch.cat([ctx_kv_exp, ctx_pe_exp], dim=-1)
        V = torch.einsum("bshc,hdc->bshd", ctx_kv_exp, wkv_b_v_full)

        Q = self.q_norm_qk(Q)
        K = self.k_norm_qk(K)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ── Build attention mask (sliding window + causal) ─────────────────
        attn_mask = self._build_attn_mask(bsz, seqlen, end_pos, mask, x.device)
        attn_out = flash_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            scale=self.softmax_scale,
        )
        out = attn_out.transpose(1, 2).contiguous().flatten(2)
        return self.wo(out)

    def _build_attn_mask(
        self,
        bsz: int,
        seqlen: int,
        end_pos: int,
        external_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compose (sliding window + causal) mask, optionally combine with
        an externally supplied padding mask.  Returns a mask of shape
        ``(bsz, 1, seqlen, end_pos)`` (or the same with more heads after
        SDPA broadcasts it).
        """
        if self.window is None or self.window <= 0:
            base = external_mask
        else:
            window = min(self.window, end_pos)
            i = torch.arange(end_pos, device=device)
            j = torch.arange(end_pos, device=device)
            causal_keep = j[None, :] <= i[:, None]  # (end_pos, end_pos)
            window_keep = j[None, :] >= i[:, None] - (window - 1)  # (end_pos, end_pos)
            local = torch.where(causal_keep & window_keep, 0.0, float("-inf"))
            local = local[-seqlen:, :]  # (seqlen, end_pos)
            if external_mask is not None:
                # external_mask: (1, 1, seqlen, end_pos) additive
                local = local.unsqueeze(0).unsqueeze(0) + external_mask
            else:
                local = local.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, end_pos)
            base = local

        if base is None:
            return None
        if base.dim() == 4:
            return base.expand(bsz, self.n_local_heads, seqlen, end_pos)
        return base
