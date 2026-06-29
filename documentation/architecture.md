# Architecture — FusionLLM-v1

> Source: `models/fusionllm.py`

FusionLLM-v1 is a 24-layer decoder-only transformer that fuses four
architectural innovations into one model:

1. **Multi-Head Latent Attention (MLA)** — DeepSeek-V2 style low-rank KV
   compression (see [mla.md](mla.md)).
2. **Gated Delta Net (GDN)** — linear-complexity attention via the delta rule
   (see [gdn.md](gdn.md)).
3. **DeepSeekMoE** — sparse mixture-of-experts with aux-loss-free routing
   (see [moe.md](moe.md)).
4. **Multi-Token Prediction (MTP)** — auxiliary future-token heads
   (see [mtp.md](mtp.md)).

## 24-Layer Schedule

The 24 `FusionLLMBlock`s alternate between two block types on a fixed
schedule (the GDN index set is hard-coded in `FusionLLM.__init__`):

| Type | Layer indices | Attention | FFN | Count |
|------|---------------|-----------|-----|:-----:|
| MLA  | 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22 | `MultiHeadLatentAttention` | `DeepSeekMoE` | 16 |
| GDN  | 2, 5, 8, 11, 14, 17, 20, 23 | `GatedDeltaNet` | `DenseFFN` (SwiGLU) | 8 |

The GDN index set `{2, 5, 8, 11, 14, 17, 20, 23}` places a GDN block every
third layer. This interleaving maximises the strengths of both attention
mechanisms: softmax-based MLA for long-range retrieval, linear GDN for
efficient context with O(T) compute.

## Block wiring

Each `FusionLLMBlock` is a standard pre-norm residual block:

```
x → RMSNorm → Attention ─→ (+) ─→ RMSNorm → FFN ─→ (+) → out
                  ↑ residual                  ↑ residual
```

- `norm1` / `norm2` are `nn.RMSNorm(dim, eps=1e-6)` (pre-attention and pre-FFN).
- The attention sub-layer is **MLA** on the 16 MLA layers and **GDN** on the
  8 GDN layers.
- The FFN sub-layer is **DeepSeekMoE** on the MLA layers and a **dense
  SwiGLU** (`DenseFFN`: `w2(silu(w1(x)) * w3(x))`) on the GDN layers.
- `self.use_checkpoint = not is_gdn`: **only MLA layers are activation
  checkpointed**. MLA's low-rank projections produce large intermediate
  activations that benefit from checkpointing; GDN is already memory-efficient
  and is left uncheckpointed to save the recomputation cost. See
  [training.md](training.md) for the memory rationale.

## Tied input/output embeddings

`self.embed = nn.Embedding(vocab_size, dim)` and
`self.head = nn.Linear(dim, vocab_size, bias=False)`. When
`tie_embeddings=True` (default), `self.head.weight = self.embed.weight` —
the input embedding matrix and the output LM head share weights. This
removes ~49 M parameters from the stored count and is a standard
Chinchilla-style parameter saving. The MTP heads also reuse this tied
weight as their output head (see [mtp.md](mtp.md)).

## Logit softcap ±15.0

```python
def softcap(logits, cap=15.0):
    return cap * torch.tanh(logits / cap)
```

The final logits are soft-capped to ±15.0 via a `tanh` squash. This is
**load-bearing**: removing it causes early-training divergence on small
batches because the cross-entropy loss on an unbounded 64K-way
classification head can spike to extreme values before the optimizer
stabilises. The cap keeps the loss landscape bounded without hard
clipping (gradients still flow through the unclamped value, just scaled).

`logit_softcap` is read from config (default 15.0); setting it to 0 or
negative disables the cap. The same cap is applied in the MTP heads via
`softcap_ce` (see [mtp.md](mtp.md)).

## μP initialization

`muP_init(model, config)` applies Maximal Update Parameterization:

- **Gate / special parameters** (`gate`, `g_proj`, `A_log`, `dt_bias`,
  `router`, `output_head` keyword match) are **zeroed**. This makes the
  MoE router and the GDN state-update gates start neutral, so the model
  begins training in a balanced, well-conditioned state.
- **Embeddings and (tied) head** use `std = 1/sqrt(dim)` (the "input/output"
  scale under μP).
- **All other 2D matrices** use `std = 1/dim` (the "hidden" scale under μP).
- **1D parameters** (norms, biases) are left to their module default init.

μP makes the training dynamics approximately width-invariant: the same
hyper-parameters transfer across model widths, and the LR does not need
re-tuning when `dim` changes. **Always verify μP LR scaling before
changing model width** (AGENTS.md hard rule 4).

`_init_weights` runs first (standard `N(0, 0.02)` for linears/embeddings,
ones for RMSNorm), then `muP_init` overrides the 2D matrices and zeroes
the gates.

## Forward pass

`forward(tokens, start_pos=0)`:

1. Embed tokens → `x` of shape `(B, T, dim)`.
2. Assert `T <= max_seq_len`.
3. For each layer: if `layer.use_checkpoint` (MLA layers), wrap in
   `torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)`;
   otherwise call `layer(x)` directly (GDN layers).
4. Apply final `RMSNorm`.
5. Project through the tied head.
6. Apply `softcap(logits, cap=logit_softcap)` if enabled.

`forward_with_hidden(tokens, start_pos=0)` is identical **except** it does
**not** activation-checkpoint (it runs every layer plain) and returns
`(logits, hidden)` — the final-norm hidden state is consumed by the MTP
heads as their input stream.

## Active vs stored parameters

- **Active: ~415.6 M** — the parameters used per forward pass (top-2 of 8
  routed experts per token + 1 shared expert per MLA layer).
- **Stored: ~868.6 M** — all parameters held in memory (all 8 routed
  expert weight matrices per MLA layer must be stored even though only 2
  are active per token).

The difference is entirely the MoE routed experts. See
[Parameter Breakdown in the README](../README.md#parameter-breakdown)
for the per-component table.