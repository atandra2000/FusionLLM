# FusionLLMBlock wiring & top-level model

> Source: `models/fusionllm.py` (~156 lines; reference exact line ranges when
> pointing at code, per AGENTS.md hard rule 5 — note the AGENTS.md "~700
> lines" figure is stale; the current file is ~156 lines after cleanup.)

This document covers the top-level `FusionLLM` class and the
`FusionLLMBlock` switch. See [architecture.md](architecture.md) for the
24-layer schedule, μP init, logit softcap, and tied embeddings.

## `FusionLLMBlock` — the MLA/GDN switch

`FusionLLMBlock.__init__(config, layer_idx, is_gdn)` is the single place
that decides whether a layer is an **MLA + MoE** layer or a **GDN + dense
SwiGLU** layer:

```python
if is_gdn:
    self.attn = GatedDeltaNet(config, layer_idx=layer_idx)
    self.ffn  = DenseFFN(config["dim"], config["inter_dim"])
else:
    self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
    self.ffn  = DeepSeekMoE(config)
```

`self.use_checkpoint = not is_gdn` — MLA layers are activation
checkpointed, GDN layers are not (see [architecture.md](architecture.md)
for the memory rationale).

> **AGENTS.md hard rule 2:** *Always read `models/__init__.py` before
> answering routing questions* — the `FusionLLMBlock` switch is the
> single source of truth for which attention/FFN a given layer index uses.
> The schedule is `{2, 5, 8, 11, 14, 17, 20, 23}` → GDN; all others → MLA.

`forward(x)` is the standard pre-norm residual:

```python
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
return x
```

## `DenseFFN` — the GDN-layer FFN

```python
class DenseFFN:
    w1: Linear(dim, inter_dim, bias=False)
    w2: Linear(inter_dim, dim, bias=False)
    w3: Linear(dim, inter_dim, bias=False)
    forward(x) = w2(silu(w1(x)) * w3(x))
```

A standard SwiGLU. Per-layer parameters: **4,718,592**
(`2 * dim * inter_dim + dim * inter_dim = 3 * 768 * 2048`, all bias-free).

## `softcap` helper

```python
def softcap(logits, cap=15.0):
    return cap * torch.tanh(logits / cap)
```

Applied to the main model's logits and (via `softcap_ce` in `mtp.py`) to
the MTP head logits. See [architecture.md](architecture.md) for why the
cap is load-bearing.

## `FusionLLM` — the full model

### `__init__(config)`
- Build `embed = Embedding(vocab_size, dim)` and `head = Linear(dim, vocab_size, bias=False)`.
- If `tie_embeddings`, tie `head.weight = embed.weight`.
- Build the 24 `FusionLLMBlock`s from the fixed GDN index set.
- Build the final `norm = RMSNorm(dim, eps=1e-6)`.
- Read `logit_softcap` (default 15.0).
- `_init_weights()` then `muP_init(self, config)` if `config["muP"]` is
  truthy (default True).

### `forward(tokens, start_pos=0)` — the training path
- Embed → for each layer (checkpointed on MLA, plain on GDN) → final norm
  → head → softcap.
- Asserts `T <= max_seq_len`.

### `forward_with_hidden(tokens, start_pos=0)` — the MTP path
- Same as `forward` but **no activation checkpointing** (every layer runs
  plain) and returns `(logits, hidden)` where `hidden` is the final-norm
  output. The MTP heads consume `hidden` as their input stream
  (see [mtp.md](mtp.md)).

### `get_moe_layers()`
Returns `[layer.ffn for layer in self.layers if not layer.is_gdn]` — used
by `Trainer` to run `update_gate_bias` and the load-balance diagnostic on
every MoE layer.

### `build_fusionllm(config)`
Thin factory: `return FusionLLM(config)`.

## Adding a new block type (from SKILLS.md Skill 2)

1. Add the block class in `models/mla.py` (or a new module).
2. Register it in `FusionLLMBlock.__init__`'s `if is_gdn: ... else: ...`
   switch.
3. Update the GDN index set in `FusionLLM.__init__` to schedule the new
   type.
4. Add a unit test in `tests/test_models.py` matching the existing 37
   tests.