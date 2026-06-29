# DeepSeekMoE — Mixture-of-Experts

> Source: `models/moe.py`

The MoE block implements the DeepSeek-style sparse expert architecture
with **auxiliary-loss-free routing**. It is used as the FFN on every MLA
layer (16 layers total).

## Configuration (from `FROZEN_CONFIG`)

| Key | Value | Meaning |
|-----|-------|---------|
| `dim` | 768 | model hidden dim |
| `n_routed_experts` | 8 | number of routed experts |
| `n_shared_experts` | 1 | always-on shared expert |
| `n_activated_experts` | 2 | top-k experts activated per token |
| `moe_inter_dim` | 2048 | per-expert SwiGLU intermediate dim |
| `route_scale` | 1.0 | scaling factor applied after softmax normalization |

Per-layer parameters: **42,473,480** (all 9 experts stored; only top-2
routed + shared active per token).

## Expert structure

Each routed expert and the shared expert is a **SwiGLU MLP**:

```python
h = (W_gate · x) ⊙ SiLU(W_up · x)        # gated activation
out = W_down · h                          # (inter_dim → dim)
```

i.e. `SwiGLUExpert(dim, inter_dim)` = `w2(silu(w1(x)) * w3(x))` with
`w1, w3: 768 → 2048` and `w2: 2048 → 768`. All linears are bias-free.

## Aux-loss-free biased-sigmoid routing (`_route`)

Instead of the standard softmax-with-load-balance-loss scheme, DeepSeekMoE
uses a **biased sigmoid gate**:

1. `logits = gate(x)` — `nn.Linear(dim, n_routed_experts, bias=True)`. The
   gate bias is zero-initialized; the gate weight is `N(0, 0.006)` for a
   near-uniform initial routing distribution.
2. `scores = sigmoid(logits)` — a per-expert probability in `[0,1]`
   (not a softmax over experts).
3. `weights, indices = topk(scores, k=n_activated_experts)` — pick the
   top-2 experts per token.
4. `weights = weights / (weights.sum(-1, keepdim=True) + 1e-10) * route_scale`
   — normalize the selected scores so they sum to `route_scale` (1.0).
   This re-normalization is what makes the routing scale-invariant.

Because the gate is a **sigmoid** (not a softmax), the experts are not in
direct competition — a token can have high affinity for multiple experts.
The **bias** on the gate is what provides load balancing (see below).

## Bias adaptation (`update_gate_bias`)

The gate bias is updated **every `bias_update_every` steps** (default 10,
driven by `Trainer.optimizer_step`) using a small fixed step:

```
bias[over] -= speed      # experts receiving too much load → bias down
bias[under] += speed      # experts receiving too little load → bias up
```

where `over = counts > avg * 1.10` and `under = counts < avg * 0.90`
(`counts` is the per-expert token count from the last forward, via
`_last_indices`). The default `speed=1e-3`.

This **shifts routing decisions toward underutilized experts without an
auxiliary load-balancing loss** — the primary language modeling objective
is not disturbed by a balancing term. The bias moves gradually, so expert
specialization is encouraged rather than forced.

## Dispatch (scatter-gather, `_dispatch_scatter_gather`)

Token-to-expert dispatch is implemented with a scatter-gather pattern
(no Triton dependency):

1. Flatten `(T, topk, n_experts)` indices and weights.
2. Build `flat_token_ids = arange(T).repeat_interleave(topk)` so each
   token's k selections know their source row.
3. `argsort` the flattened expert indices to group tokens by expert.
4. `unique_consecutive` to find expert boundaries and per-expert counts.
5. For each expert with tokens: `expert_out = experts[idx](x[token_ids])`
   and `index_add_` the weighted output back into the token buffer.

This is a Python-loop-over-experts fallback; it is correct and avoids the
Triton dependency, at the cost of some kernel-launch overhead.

## Shared expert

`shared_expert = SwiGLUExpert(dim, moe_inter_dim)` (always active, not
routed). The final output is `y_routed + y_shared`. The shared expert
captures the "common" sub-circuit that every token needs, leaving the
routed experts to specialize.

## Forward state

`forward` stores `self._last_indices` and `self._last_weights` (detached)
on each call. These feed `update_gate_bias` and `get_load_balance_loss`.
`get_load_balance_loss` computes the standard auxiliary loss
`(f * P).sum() * n_experts` (where `f` is the per-expert frequency and
`P` is the mean per-expert routing weight) — it is available for
diagnostics but is **not added to the training loss** in
`Trainer.train_step` by default (only a tiny `balance_loss_alpha = 1e-4`
weight is used, and the loss is the aux-loss-free surrogate).