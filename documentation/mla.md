# Multi-Head Latent Attention (MLA)

> Source: `models/mla.py`

MLA replaces standard multi-head attention with **low-rank KV compression
via latent projections**, as introduced in DeepSeek-V2. It reduces the KV
cache size and the attention compute cost while keeping full attention
quality.

## Configuration (from `FROZEN_CONFIG`)

| Key | Value | Meaning |
|-----|-------|---------|
| `dim` | 768 | model hidden dim |
| `n_heads` | 12 | query heads |
| `n_kv_groups` | 8 | KV groups (GQA 12:8) |
| `q_lora_rank` | 192 | query low-rank bottleneck |
| `kv_lora_rank` | 96 | KV latent compression rank |
| `qk_nope_head_dim` | 64 | per-head non-positional dim |
| `qk_rope_head_dim` | 32 | per-head RoPE-bearing dim (hard-coded in the decoupled RoPE path — **changing it requires retraining from scratch**) |
| `v_head_dim` | 64 | per-head value dim |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `max_seq_len` | 4096 | cached RoPE length |

Per-layer parameters: **1,155,616**.

## Projection structure

### Query path
`x (B,T,768)` → `wq_a` → `(B,T,192)` → `q_norm` (RMSNorm) → `wq_b` →
`(B,T,12,96)` → split into `q_nope (B,T,12,64)` and `q_pe (B,T,12,32)`.
`q_pe` is rotated by `RotaryEmbedding(head_dim=32)`.

### Key/Value path
`x (B,T,768)` → `wkv_a` → `(B,T,128)` → split into
`kv_latent (B,T,96)` and `k_pe_raw (B,T,32)`.
- `kv_latent` → `kv_norm` (RMSNorm) → `kv_normed (B,T,96)`.
- `k_pe_raw` → RoPE (applied on a `(B,T,1,32)` view then squeezed back) →
  `k_pe (B,T,32)`.

`wkv_b` has weight shape `(n_kv_groups, qk_nope_head_dim + v_head_dim,
kv_lora_rank) = (8, 128, 96)`. It is split into `wkv_b_k (8,64,96)` and
`wkv_b_v (8,64,96)`.

## Absorption trick

To avoid materializing the full per-head K and V, MLA uses the
**absorption trick** (`W_Q @ W_UK.T` style):

- `wkv_b_k` and `wkv_b_v` are **expanded per query head** by indexing with
  `_kv_group_for_q` (a precomputed buffer mapping each of the 12 query
  heads to one of the 8 KV groups — this implements the GQA 12:8 grouping).
- `q_nope_proj = einsum("bthd,hdc->bthc", q_nope, wkv_b_k_q)` — the query
  non-positional component is multiplied with the absorbed key weight
  *before* attention, so the explicit full K is never materialized on the
  inference path.
- `v = einsum("btc,hdc->bthd", kv_normed, wkv_b_v_q)` — value is produced
  per query head from the compressed KV latent.

> **Pitfall (AGENTS.md / SKILLS.md):** the absorption trick must stay on
> the **inference** path only. **Training needs material K/V for correct
> gradients** — do not "optimize" the training forward by skipping the
> explicit projections.

## QK-Norm

Before attention, `q_concat` and `k_concat` are each passed through a
shared `RMSNorm` (`q_norm_qk` / `k_norm_qk`, both over
`kv_lora_rank + qk_rope_head_dim = 128`). This is the QK-Norm stabilization
trick: normalizing Q and K per-head before the dot product prevents the
attention logits from blowing up.

## Attention compute (Flash Attention 2 with SDPA fallback)

`mla.py` imports `flash_attn_func` defensively:

```python
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
```

- **If `flash_attn_func` is available**: transpose Q/K/V from
  `(B, n_heads, T, head_dim)` to `(B, T, n_heads, head_dim)` and call
  `flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=True)`. FA2
  gives ~40–50% speedup on attention forward/backward and lower memory.
- **Otherwise**: fall back to
  `F.scaled_dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)`.

In both branches the output is transposed back and reshaped to
`(B, T, n_heads * v_head_dim)` before the final output projection `wo`.

## RoPE caching

`RotaryEmbedding` caches `cos` / `sin` tables keyed by sequence length and
rebuilds them only when the requested length exceeds `_cached_len`. The
cache is non-persistent (`register_buffer(..., persistent=False)`). RoPE
is applied with the rotate-half formulation
(`x * cos + x_rot * sin` where `x_rot` is the rotate-pair of `x`).