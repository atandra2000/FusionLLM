# Training — dual optimizer, WSD scheduler, checkpoints, benchmark

> Source: `training/` (`trainer.py`, `optimizer.py`, `scheduler.py`,
> `checkpoint.py`, `validation.py`, `benchmark.py`, `data_loader.py`)

## Dual optimizer (NorMuon + CautiousAdamW)

> **AGENTS.md hard rule 3 (load-bearing):** *Always preserve the
> dual-optimizer split: NorMuon for 2D matrices, CautiousAdamW for
> norms/biases/embeddings. Switching them breaks training.*

FusionLLM uses **two optimizers in parallel**, each on a different
parameter group. The split is enforced in
`training/optimizer.py:build_optimizers`.

### NorMuon (`NorMuon`, lr 0.02)

For **2D matrix parameters** (attention projections, FFN weights, MoE
expert weights). NorMuon is orthogonalized Adam with per-row RMS
normalization:

1. Standard Adam moment estimates (`exp_avg`, `exp_avg_sq`) with
   `betas=(0.95, 0.95)` by default (momentum configurable via
   `muon_momentum`).
2. Bias-corrected update `update = exp_avg / bias_corr1 / (sqrt(exp_avg_sq) / sqrt(bias_corr2) + eps)`.
3. **Per-row RMS normalization** for 2D params:
   `row_rms = ||update||_2 / sqrt(d) + eps; update = update / (row_rms + eps)`.
   This rescales each row of the update to unit RMS, which is the
   "orthogonalization" step that gives Muon/NorMuon its stability on
   matrix parameters.
4. **Cautious weight decay**: `p *= 1 - lr * wd * mask` where
   `mask = (grad * p).sign() == 1` — decay is only applied where the
   gradient and the parameter agree in sign (i.e. the update is
   "agreeing" with the loss landscape), avoiding decay that fights an
   active gradient.

### CautiousAdamW (`CautiousAdamW`, lr 3e-4)

For **1D parameters** (RMSNorm γ, biases, embeddings, the tied head, and
all gate/special parameters: `A_log`, `dt_bias`, `D`, `gate.bias`, any
param whose name matches the `exclude_patterns` tuple). Standard AdamW
with the same cautious weight-decay mask as NorMuon (applied only to 2D
params in this group; 1D params get plain `p *= 1 - lr * wd`).

The conservative lr (3e-4 vs NorMuon's 0.02) reflects that these
parameters (norms, biases, embeddings) are sensitive to large updates.

### `build_optimizers(model, ...)`

Splits parameters by `name` and `ndim`:

- `exclude_patterns = ("embed", "head", "norm", "bias", "gate.bias",
  "proj", "A_log", "dt_bias", "D")` — any parameter whose name contains
  one of these substrings goes to CautiousAdamW.
- Of the rest, params with `ndim >= 2` go to NorMuon; everything else to
  CautiousAdamW.
- Tied parameters are de-duplicated by `id(p)` so the tied
  embedding/head is only counted once.
- Returns `(muon_opt, adamw_opt)`; `muon_opt` is `None` if no matrix
  params were found.

### Step order in `Trainer.optimizer_step`

1. `clip_grad_norm_(train_model.parameters(), 1.0)` — global norm clip.
2. `muon_opt.step()` (if not None).
3. `adamw_opt.step()`.
4. `scaler.update()` (the GradScaler is disabled in BF16 — kept for API
   compatibility).
5. `train_model.zero_grad(set_to_none=True)`.
6. Every `bias_update_every` (10) steps: `moe.update_gate_bias(speed=1e-3)`
   on every MoE layer (see [moe.md](moe.md)).
7. `scheduler.step()` — advances the WSD scheduler (drives the AdamW lr;
   NorMuon's lr is held constant at 0.02).

> **Pitfall:** the scheduler is wired to `adamw_opt` only (see
> `Trainer.__init__`). NorMuon's lr is **not** scheduled — it stays at
> `muon_lr` for the whole run. This is intentional: the matrix updates
> rely on the orthogonalization step for scale, not on LR annealing.

## WSD scheduler (`WSDScheduler`)

Warmup-Stable-Decay, from `_LRScheduler`:

| Phase | Fraction of `total_steps` | LR factor |
|-------|---------------------------|-----------|
| Warmup | `warmup_frac` (default 0.01 → 1%) | linear `0 → 1.0` |
| Stable | `stable_frac` (default 0.84 → 84%) | constant `1.0` |
| Decay  | remainder (default 15%) | linear `1.0 → min_lr_ratio` (default 0.1) |

- `total_steps` default 63,400 (8.31B tokens / 131,072 tokens per step).
- `decay` supports `"linear"` (default) or `"cosine"`.
- `get_lr(step)` returns `base_lr * factor` for each base LR in
  `base_lrs`.
- After `total_steps`, the factor is held at `min_lr_ratio`.
- `step_optimizers()` steps the primary optimizer and propagates the
  computed LR to any additional optimizers in `self._optimizers` (used
  when wiring the scheduler across multiple param groups).

> **SKILLS.md pitfall:** the stable phase must be ≥80% of total steps;
> the 84% default is tuned for 8B-token runs.

## Precision & checkpoints

- **Precision:** BF16 autocast (`torch.bfloat16`), native (no GradScaler
  needed). `Trainer.dtype = torch.bfloat16` when `config["dtype"] == "bf16"`.
- **Checkpoint format:** `safetensors` for model weights (cast to BF16),
  `torch.save` for optimizer state (`{"muon": ..., "adamw": ...}`),
  JSON for metadata (`step`, `token_count`, `best_loss`, scheduler state,
  model config).
- **Save:** `save_checkpoint(...)` writes to
  `checkpoints/pretrain/<tag>/` (`model.safetensors`, `optimizer.pt`,
  `metadata.json`). Falls back to `torch.save` if `safetensors` is not
  installed.
- **Load:** `load_checkpoint(...)` restores model, optimizers (muon +
  adamw), and scheduler state. Returns the metadata dict.
- **Latest discovery:** `find_latest_checkpoint(save_dir)` scans
  subdirectories, reads each `metadata.json`'s `step`, and returns the
  directory with the highest step.
- **Retention:** `_cleanup_old_checkpoints` keeps only `max_keep` (default
  3) checkpoints, removing the oldest `step_*` directories.
- **Trainer.save / Trainer.load:** thin wrappers that pass the current
  `muon_opt`, `adamw_opt`, `scheduler`, `step`, `token_count`, `best_loss`.
  `Trainer.load` also restores `self.step`, `self.global_step`,
  `self.token_count`, `self.best_loss`.

### Resume from a safetensors checkpoint (SKILLS.md Skill 5)

```python
from safetensors.torch import load_file
state = load_file("checkpoints/fusionllm_step_50000.safetensors")
model.load_state_dict(state)
# or via the Trainer:
trainer.train(resume_from="latest")
```

## Validation (`validation.py`)

- **Synthetic data:** `generate_synthetic_batch` produces random
  `(tokens, targets)` in `[0, vocab_size)`. The real run uses real data,
  but the validation harness is synthetic (per `eval_synthetic: true`).
- `compute_validation_loss(model, ...)` runs `num_batches` (default 8)
  forward passes in `@torch.no_grad()` eval mode, sums cross-entropy over
  all tokens, and returns `{"loss", "ppl", "n_tokens"}`.
- `validate_forward_shape(model, ...)` asserts the model output is
  `(batch_size, seq_len, 64000)` and prints the shape.
- `Trainer.train_epoch` calls `compute_validation_loss` every
  `eval_interval` (5000) steps and saves a `"best"` checkpoint when the
  validation loss improves.

## Benchmark (`benchmark.py`)

- Constructs the model with `mtp_depth=0` (MTP disabled for the
  benchmark) and the A100-optimized config (`micro_batch_size=4`,
  `gradient_accumulation_steps=8`, `use_compile=True`,
  `compile_mode="reduce-overhead"`, `use_checkpoint_per_layer=True`).
- 10 warmup steps (for `torch.compile` graph capture), then `steps`
  (default 100) measured steps with synthetic random data.
- Prints total tokens, elapsed seconds, throughput (tokens/sec), and
  estimated training time for 8.31B tokens.

**Measured throughput on a single A100 80GB:**

| Config | Throughput | Est. training time (8.31B tok) |
|--------|-----------|-------------------------------|
| BF16 + `torch.compile` + Flash Attention 2 | 20,000 – 28,000 tok/s | 3.4 – 4.8 days |
| BF16 + `torch.compile` (no FA2) | ~15,000 – 18,000 tok/s | ~5.3 – 6.4 days |

## Async data loader (`data_loader.py`)

`AsyncDataLoader` wraps a base iterator with:
- **Pinned-memory prefetching** (`pin_memory=True`): CPU tensors are
  moved to pinned memory before transfer.
- **Non-blocking GPU transfer** (`to(device, non_blocking=True)`): H2D
  copy overlaps with compute.
- **Throughput benchmarking** (`benchmark(num_batches)`): measures
  `batches_per_sec`, `tokens_per_sec`, `elapsed_sec` for data-loading
  isolation.

`Trainer.train_epoch` wraps the input `data_iter` in an `AsyncDataLoader`
before the training loop and (at step 0) logs the data-loading throughput
to detect a data-feeding bottleneck.