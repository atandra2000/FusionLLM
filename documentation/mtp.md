# Multi-Token Prediction (MTP)

> Source: `models/mtp.py`

MTP augments the next-token prediction objective with **auxiliary
future-token prediction heads**, following the approach validated in
recent large-scale pre-training studies (e.g. DeepSeek-V3). It provides
extra supervision signal without changing the main model.

## Configuration (from `FROZEN_CONFIG`)

| Key | Value | Meaning |
|-----|-------|---------|
| `mtp_depth` | 2 | number of auxiliary heads (depth 2 and depth 3) |
| `mtp_loss_weight_1` | 0.10 | loss weight for depth-2 head (λ₂) |
| `mtp_loss_weight_2` | 0.05 | loss weight for depth-3 head (λ₃) |
| `mtp_softcap` | True | apply logit softcap in the MTP cross-entropy |
| `mtp_softcap_value` | 15.0 | softcap value (matches the main model) |
| `mtp_inter_dim` | 2048 | SwiGLU FFN dim in the shared MTP block |

Additional MTP-specific parameters: **~14,109,248** (≈2.46 M per the
README's older accounting; the exact count is verified by
`test_mtp_total_params`).

## Architecture

`MultiTokenPrediction` wraps the main model and adds `mtp_depth`
`MTPModule` heads. All heads share:

- A single `MTPTransformerBlock` instance per head (pre-norm RMSNorm +
  `MultiHeadLatentAttention` + `SwiGLUExpert` FFN).
- The main model's **tied output head** (`main_model.head`) as their
  output projection (set via `set_output_head`).

### Depth-d head (`MTPModule`)

For depth `d` (1-indexed: depth-1 is the first auxiliary head, predicting
`t+2`; depth-2 predicts `t+3`):

- `norm_h` (RMSNorm) over the previous hidden state `hidden[t]`.
- `norm_e` (RMSNorm) over the target embedding `embed[t+d-1]`.
- `proj = Linear(2*dim, dim)` fuses the two into a single hidden vector.
  (For `depth >= 2`, `proj_aux` is used; for `depth == 1`, `proj` is used.
  The two-projection structure lets deeper heads reuse the same fusion
  pattern with a fresh parameter set.)
- `block = MTPTransformerBlock` applies one transformer layer (MLA +
  SwiGLU FFN) to the fused hidden state.
- `norm_out` + the shared `output_head` produce the logits for token
  `t+d+1`.
- The new hidden state `h` is returned and feeds the next depth head.

### `forward(tokens, start_pos=0)`

1. Run the main model via `forward_with_hidden` to get
   `(main_logits, main_hidden)` — note this path is **not** checkpointed
   (see [architecture.md](architecture.md)).
2. For each depth `d` from 1 to `mtp_depth`:
   - `usable = T - depth - 1` (the head can only predict tokens that have
     a real target at `t+depth+1`).
   - `h_in = prev_hidden[:, :usable]` (the previous depth's output, or
     `main_hidden` for depth 1).
   - `target_emb = embed(tokens[:, depth:depth+usable])`.
   - `targets = tokens[:, depth+1:depth+1+usable]`.
   - `(logits, new_hidden) = mtp(h_in, target_emb)`.
   - Append `(logits, targets, weight=mtp_loss_weights[d])` to outputs.
   - `prev_hidden = new_hidden` for the next depth.
3. Return `(main_logits, mtp_outputs)`.

## Loss integration (`compute_mtp_loss`)

The MTP loss is a **weighted sum** of per-depth cross-entropies, added to
the main next-token loss in `Trainer.train_step`:

```
total_loss = sum_d  λ_d · softcap_ce(logits_d, targets_d)
```

- `softcap_ce` applies the same `±15.0` `tanh` softcap as the main model
  before `F.cross_entropy` (when `mtp_softcap=True`). This keeps the
  auxiliary loss bounded the same way the main loss is.
- The small weights `(0.10, 0.05)` ensure the auxiliary objectives **do
  not dominate** the primary language modeling objective. They provide
  gradient signal that encourages the hidden states to be predictive of
  the near future, which improves representation quality.

`Trainer.train_step` computes `loss = main_loss + mtp_loss` (the MTP loss
already includes its own weights).