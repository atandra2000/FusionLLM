# Gated Delta Net (GDN)

> Source: `models/gdn.py`

GDN is a **linear-complexity attention** mechanism based on a gated
delta-rule state update. Unlike softmax attention, GDN maintains a
recurrent state updated via an outer-product (delta-rule) operation,
achieving **O(T) memory and compute** with respect to sequence length
(rather than O(T²) for softmax attention).

## Configuration (from `FROZEN_CONFIG`)

| Key | Value | Meaning |
|-----|-------|---------|
| `dim` | 768 | model hidden dim |
| `gdn_d_inner` | 1024 | inner (projected) width |
| `gdn_d_state` | 32 | recurrent state dim per head |
| `gdn_d_conv` | 4 | causal depthwise conv kernel size |
| `gdn_headdim` | 32 | per-head width → `n_heads = d_inner // headdim = 32` |
| `gdn_chunk_size` | 64 | chunk size for the delta-rule recurrence |

Per-layer parameters: **8,688,704**.

> **Pitfall (SKILLS.md Skill 3):** `chunk_size` (default 64) affects
> numerical stability of the recurrent state. Changing it requires
> re-tuning `sigmoid` vs `snake` gating by ablation (the paper uses
> snake). Verify peak memory before increasing it.

## Processing pipeline

`forward(x)` where `x` is `(B, T, 768)`:

1. **Input projection:** `in_proj` maps `768 → 6*1024`, then split into six
   streams `z, x, b, c, dt, g` (each `(B, T, 1024)`).
2. **Causal depthwise 1D conv** (`conv1d`, kernel 4, groups 1024, causal
   padding) applied to the `x` stream, then `silu`. The conv is causal —
   padding is on the left so position `t` only sees positions `<= t`.
3. **Per-head projections:**
   - `B_proj = b_proj(x_conv) → (B, T, 32, 32)` (the "key"-like stream)
   - `C_proj = c_proj(x_conv) → (B, T, 32, 32)` (the "query"-like stream)
   - `dt = softplus(dt_proj(x_conv) + dt_bias) → (B, T, 32)` (per-head
     decay/gate, bias-initialized `U(0.001, 0.1)`)
   - `g = sigmoid(g_proj(x_conv)) → (B, T, 1024)` (output gate)
   - `v = x_conv.view(B, T, 32, 32)` (the "value" stream, reshaped to
     `(n_heads, headdim) = (32, 32)`)
4. **Decay:** `A = -exp(A_log)` where `A_log` is initialized to
   `log(arange(1, n_heads+1).repeat_interleave(d_state))`, giving each head
   a distinct negative decay. `decay = sigmoid(dt * A)`.
5. **State update (delta rule):** see `_chunked_delta_rule` below.
6. **Skip connection:** `y = y + v * D` where `D` is a learned per-head
   scalar (initialized to 1).
7. **Output gating & projection:**
   `out = out_proj(y.reshape(B, T, d_inner) * g * silu(z))`.

The final modulation `g * silu(z)` is the **snake/sigmoid gating**: `silu(z)`
acts as a non-linear activation and `sigmoid(g)` is a multiplicative gate.

## Chunked delta-rule recurrence (`_chunked_delta_rule`)

This is the core linear-attention state update, run in **FP32** for
numerical stability (the recurrent state is repeatedly updated, so
accumulated BF16 rounding error would degrade quality):

```
state = 0                                        # (B, n_heads, headdim, d_state) float32
for chunk in chunks of size chunk_size:
    for t in chunk:
        k_t = normalize(B_proj[t], dim=-1)       # unit-norm key
        v_t = v[t]
        dec_t = decay[t]                         # sigmoid(dt * A)
        write = outer(v_t, k_t)                   # (headdim, d_state)
        state = dec_t * state + write             # delta-rule update
        y[t] = state @ C_proj[t]                  # read
```

- `k` is **L2-normalized** (`F.normalize(B.float(), dim=-1, eps=1e-6)`)
  before being used as the key in the outer product. This keeps the state
  update scale-stable.
- The chunk loop is purely for **GPU utilization** (vectorizing within a
  chunk would be a future optimization; the current per-`t` inner loop is
  the naive-but-correct reference). Chunking does not change the math.
- The output tensor `y` is pre-allocated in FP32 and cast back to the
  input dtype at the end.

## No-weight-decay parameters

`A_log`, `D`, and `dt_bias` are marked `._no_weight_decay = True` so the
optimizer builder excludes them from weight-decay-bearing groups (see
[training.md](training.md)).