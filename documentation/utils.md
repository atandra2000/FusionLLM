# Utils — checkpoint, scheduler, validation, benchmark, async loader

> Source: `training/checkpoint.py`, `training/scheduler.py`,
> `training/validation.py`, `training/benchmark.py`,
> `training/data_loader.py`

This is a quick-reference companion to [training.md](training.md),
focused on the utility modules. See [training.md](training.md) for the
full dual-optimizer / WSD / benchmark narrative.

## `checkpoint.py`

### `save_checkpoint(model, muon_opt, adamw_opt, scheduler, step, token_count, best_loss, save_dir, max_keep, tag)`
- Creates `save_dir/<tag>/` (tag defaults to `f"step_{step}"`).
- **Model:** `model.state_dict()` cast to `bfloat16` →
  `model.safetensors` via `_save_safetensors` (falls back to
  `torch.save(..., .pt)` if `safetensors` is not installed).
- **Optimizers:** `{"muon": muon_opt.state_dict(), "adamw": adamw_opt.state_dict()}`
  → `optimizer.pt` via `torch.save`.
- **Metadata:** `{step, token_count, best_loss, scheduler.state_dict(), model_config}`
  → `metadata.json` (passed through `_make_serializable` which recursively
  converts non-JSON-serializable objects to `str`).
- Calls `_cleanup_old_checkpoints(save_dir, max_keep)` to remove old
  `step_*` directories beyond `max_keep`.

### `load_checkpoint(model, muon_opt, adamw_opt, scheduler, load_dir, device, strict)`
- Loads `model.safetensors` (or `.pt` fallback) into the model.
- Loads `optimizer.pt` (via `torch.load(..., weights_only=True)`) into
  `muon_opt` and `adamw_opt` if present.
- Loads `metadata.json` and restores `scheduler.load_state_dict(...)` if
  present.
- Returns the metadata dict.

### `find_latest_checkpoint(save_dir)`
Scans subdirectories of `save_dir`, reads each `metadata.json`'s `step`,
and returns the directory with the highest step (or `None`).

### `_save_safetensors` / `_load_safetensors`
Thin wrappers around `safetensors.torch.save_file` / `load_file` with a
`torch.save`/`torch.load` fallback. Used so the checkpoint code works
without `safetensors` installed (BF16 is still preserved on the
`torch.save` path).

### `_make_serializable`
Recursively walks a dict/list and converts anything that is not
`int|float|str|bool|None` to `str(...)`. Used so the metadata JSON can
hold the scheduler state dict (which may contain tensors) without
crashing `json.dump`.

### `_cleanup_old_checkpoints`
Sorts `step_*` directories by step number descending and `rmtree`s
everything beyond `max_keep`.

## `scheduler.py` — `WSDScheduler`

Subclass of `torch.optim.lr_scheduler._LRScheduler`. See
[training.md](training.md#wsd-scheduler-wsdscheduler) for the phase
table. Implementation notes:

- Accepts a single optimizer **or** a list/tuple of optimizers
  (`self._optimizers`). The primary optimizer is passed to the parent
  `_LRScheduler.__init__`; the others are stepped via
  `step_optimizers()` which propagates `get_lr()[0]` to their param
  groups.
- `get_lr()` returns `[base_lr * factor for base_lr in self.base_lrs]`
  where `factor` depends on the phase (linear warmup, constant stable,
  linear/cosine decay to `min_lr_ratio`, clamped at `min_lr_ratio` after
  `total_steps`).
- `decay` is asserted to be `"linear"` or `"cosine"`.

## `validation.py`

- `generate_synthetic_batch(batch_size, seq_len, vocab_size, device)`:
  returns random `(tokens, targets)` in `[0, vocab_size)`.
- `compute_validation_loss(model, batch_size, seq_len, vocab_size,
  num_batches, device)` (`@torch.no_grad`): runs `num_batches` forward
  passes in eval mode, sums `F.cross_entropy(..., reduction="sum")`,
  returns `{"loss": avg, "ppl": exp(avg), "n_tokens": total}`. Toggles
  `model.train()` back on at the end.
- `validate_forward_shape(model, batch_size, seq_len, device)`
  (`@torch.no_grad`): asserts the model output is
  `(batch_size, seq_len, 64000)` and logs the shape.

## `benchmark.py`

CLI entry point (`python training/benchmark.py --steps 100`). See
[training.md](training.md#benchmark-benchmarkpy) for the full narrative.
Key points:

- `get_config()` returns the A100-optimized config with `mtp_depth=0`
  (MTP disabled for the benchmark), `micro_batch_size=4`,
  `gradient_accumulation_steps=8`, `use_compile=True`,
  `compile_mode="reduce-overhead"`, `use_checkpoint_per_layer=True`,
  `wandb_enabled=False`.
- `create_mock_data_iter(batch_size, seq_len, vocab_size, device)`: an
  infinite generator yielding `(tokens, targets)` with
  `targets = tokens.clone()`.
- `benchmark(steps)`: builds the model on CUDA (or CPU with a warning),
  optionally `torch.compile`s it, runs 10 warmup steps (graph capture)
  followed by `steps` measured steps, and prints total tokens, elapsed
  seconds, throughput, and estimated training time for 8.31B tokens.
- Uses `torch.cuda.amp.autocast(dtype=torch.bfloat16)` on CUDA and a
  disabled `torch.autocast` on CPU.

## `data_loader.py` — `AsyncDataLoader`

Wraps a base `(tokens, targets)` iterator. See
[training.md](training.md#async-data-loader-data_loaderpy) for the
narrative. Implementation notes:

- `__init__` stores `data_iter, device, prefetch_factor, num_workers,
  pin_memory, batch_size, seq_len, vocab_size` and a `None`
  `_prefetch_queue`.
- `_create_prefetch_iterator()` is a generator that pulls from
  `self.data_iter`; if `pin_memory` and the tensor is on CPU, it calls
  `.pin_memory()` on both tokens and targets before yielding.
- `__iter__` pulls from the prefetch iterator and does
  `.to(device, non_blocking=True)` on both tensors.
- `benchmark(num_batches)` runs `num_batches` iterations and returns
  `{"batches_per_sec", "tokens_per_sec", "elapsed_sec"}` where
  `tokens_per_sec = num_batches * batch_size * seq_len / elapsed`.

> Note: `prefetch_factor` and `num_workers` are stored but not currently
> used for background workers — the prefetching is the pinned-memory +
> non-blocking-transfer pattern above. They are kept on the API for
> future background-thread prefetching.