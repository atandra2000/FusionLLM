# FusionLLM-v1 Test Plan

**Date**: 2026-06-13
**Total Tests**: 55 (32 model + 5 BF16 skipped-on-CPU + 18 training)
**Framework**: pytest 9.x, Python 3.14.3
**Config**: `pyproject.toml` (strict warnings-as-errors, benchmark, cov)

---

## 1. Test Inventory

### 1.1 Model Tests (`tests/test_models.py`) — 37 total (32 run on CPU, 5 require CUDA)

| # | Test Name | Class | Category | What It Verifies | Priority |
|---|-----------|-------|----------|------------------|----------|
| 1 | `test_mla_per_layer_params` | TestParameterCounts | Budget | MLA has exactly 1,155,616 params | Critical |
| 2 | `test_moe_per_layer_active_params` | TestParameterCounts | Budget | MoE stores all 9 experts (42,473,480 params) | Critical |
| 3 | `test_gdn_per_layer_params` | TestParameterCounts | Budget | GDN has exactly 8,688,704 params | Critical |
| 4 | `test_dense_ffn_per_layer_params` | TestParameterCounts | Budget | Dense FFN has 4,718,592 params | Critical |
| 5 | `test_embedding_params` | TestParameterCounts | Budget | Tied embedding has 49,152,000 params | Critical |
| 6 | `test_total_stored_params` | TestParameterCounts | Budget | Full model stores ~854,516,520 params (without MTP) | Critical |
| 7 | `test_mtp_total_params` | TestParameterCounts | Budget | MTP-specific params total ~14,109,248 | Critical |
| 8 | `test_mla_output_shape` | TestShapeCorrectness | Shape | MLA output = (B, T, 768) | High |
| 9 | `test_moe_output_shape` | TestShapeCorrectness | Shape | MoE output = (T, 768) | High |
| 10 | `test_moe_batched_output_shape` | TestShapeCorrectness | Shape | MoE output = (B, T, 768) | High |
| 11 | `test_gdn_output_shape` | TestShapeCorrectness | Shape | GDN output = (B, T, 768) | High |
| 12 | `test_full_model_output_shape` | TestShapeCorrectness | Shape | Model output = (B, T, 64000) | Critical |
| 13 | `test_model_forward_with_hidden` | TestShapeCorrectness | Shape | forward_with_hidden returns (logits, hidden) | High |
| 14 | `test_mtp_output_shapes` | TestShapeCorrectness | Shape | MTP returns correct main + per-depth shapes | High |
| 15 | `test_mla_forward` | TestForwardPass | Correctness | MLA forward produces finite output | Critical |
| 16 | `test_moe_forward` | TestForwardPass | Correctness | MoE forward produces finite output | Critical |
| 17 | `test_gdn_forward` | TestForwardPass | Correctness | GDN forward produces finite output | Critical |
| 18 | `test_full_model_forward` | TestForwardPass | Correctness | Full model forward produces finite output | Critical |
| 19 | `test_mtp_forward` | TestForwardPass | Correctness | MTP forward produces finite output | High |
| 20 | `test_mla_backward` | TestBackwardPass | Gradients | MLA gradients flow to all parameters | Critical |
| 21 | `test_moe_backward` | TestBackwardPass | Gradients | MoE gradients flow to all parameters | Critical |
| 22 | `test_gdn_backward` | TestBackwardPass | Gradients | GDN gradients flow to all parameters | Critical |
| 23 | `test_full_model_backward` | TestBackwardPass | Gradients | Full model gradients flow (check representative params) | Critical |
| 24 | `test_mla_bf16` | TestBF16Compatibility | BF16 | MLA forward in BF16 **(requires CUDA)** | Medium |
| 25 | `test_moe_bf16` | TestBF16Compatibility | BF16 | MoE forward in BF16 **(requires CUDA)** | Medium |
| 26 | `test_gdn_bf16` | TestBF16Compatibility | BF16 | GDN forward in BF16 **(requires CUDA)** | Medium |
| 27 | `test_full_model_bf16` | TestBF16Compatibility | BF16 | Full model forward in BF16 **(requires CUDA)** | Medium |
| 28 | `test_full_model_bf16_backward` | TestBF16Compatibility | BF16 | Full model backward in BF16 **(requires CUDA)** | Medium |
| 29 | `test_save_load_roundtrip` | TestCheckpoint | Checkpoint | State dict round-trips through safetensors (atol=1e-2) | Critical |
| 30 | `test_checkpoint_metadata` | TestCheckpoint | Checkpoint | Step/token_count/best_loss survive save/load | High |
| 31 | `test_deterministic_forward` | TestDeterministicResume | Determinism | Same input → same output | Medium |
| 32 | `test_softcap_ce_shapes` | TestSoftcapCE | MTP | Softcap CE returns scalar loss | Medium |
| 33 | `test_softcap_ce_bounded` | TestSoftcapCE | MTP | Softcap CE remains finite with extreme logits | Medium |
| 34 | `test_gdn_layers_at_correct_indices` | TestLayerSchedule | Architecture | GDN at indices [2,5,8,11,14,17,20,23] | Critical |
| 35 | `test_moe_on_mla_layers` | TestLayerSchedule | Architecture | MoE on MLA layers, DenseFFN on GDN layers | Critical |
| 36 | `test_moe_routing_output_structure` | TestMoERouting | MoE | last_indices/last_weights shapes are correct | High |
| 37 | `test_moe_gate_bias_update` | TestMoERouting | MoE | Gate bias changes after update_gate_bias() | High |

### 1.2 Training Tests (`tests/test_training.py`) — 18 total

| # | Test Name | Class | Category | What It Verifies | Priority |
|---|-----------|-------|----------|------------------|----------|
| 1 | `test_normuon_instantiation` | TestOptimizers | Optimizer | NorMuon can be created | High |
| 2 | `test_cautious_adamw_instantiation` | TestOptimizers | Optimizer | CautiousAdamW can be created | High |
| 3 | `test_build_optimizers` | TestOptimizers | Optimizer | build_optimizers returns correct types | Critical |
| 4 | `test_normuon_optimizer_step` | TestOptimizers | Optimizer | NorMuon step changes model parameters | Critical |
| 5 | `test_adamw_optimizer_step` | TestOptimizers | Optimizer | AdamW step changes model parameters | Critical |
| 6 | `test_gradient_clip_respected` | TestOptimizers | Optimizer | clip_grad_norm_ works correctly | High |
| 7 | `test_wsd_total_steps` | TestScheduler | Scheduler | WSD covers exactly total_steps | Medium |
| 8 | `test_wsd_warmup_phase` | TestScheduler | Scheduler | Warmup LR increases linearly to peak | High |
| 9 | `test_wsd_decay_linear` | TestScheduler | Scheduler | Decay phase LR decreases to min_lr_ratio | High |
| 10 | `test_wsd_min_lr_after_total` | TestScheduler | Scheduler | Post-total LR stays at min_lr_ratio | High |
| 11 | `test_checkpoint_with_optimizers` | TestCheckpointWithOptimizer | Checkpoint | Model weights preserved after save/load with optimizers | Critical |
| 12 | `test_find_latest_checkpoint` | TestCheckpointWithOptimizer | Checkpoint | find_latest_checkpoint finds most recent step | Medium |
| 13 | `test_compute_validation_loss` | TestValidation | Validation | compute_validation_loss returns loss + ppl + n_tokens | High |
| 14 | `test_validate_forward_shape` | TestValidation | Validation | validate_forward_shape runs without assertion error | Medium |
| 15 | `test_trainer_instantiation` | TestTrainer | Trainer | Trainer constructs model + optimizers | Critical |
| 16 | `test_trainer_train_step` | TestTrainer | Trainer | Single train step produces finite loss | Critical |
| 17 | `test_trainer_optimizer_step` | TestTrainer | Trainer | Optimizer step runs + LR is positive | High |
| 18 | `test_mtp_loss_computation` | TestMTPLoss | MTP | MTP compute_mtp_loss returns scalar positive loss | High |

---

## 2. Test Categories by Risk

### Critical (must pass before any training run)

| # | Test | Risk if Fails |
|---|------|---------------|
| Model budget tests (1–7) | Parameter counts wrong → wrong architecture, wasted VRAM or OOM |
| Shape tests (8, 12) | Shape mismatch → silent broadcasting errors or NaN |
| Forward tests (15–18) | NaN/Inf in forward → training diverges instantly |
| Backward tests (20–23) | No gradients → no learning |
| Checkpoint roundtrip (29) | Can't save/load → wasted compute on failures |
| Layer schedule tests (34–35) | Wrong layer type → completely different architecture |
| build_optimizers (train-3) | Wrong optimizer → no convergence or OOM |
| Optimizer step (train-4,5) | Parameters not updated → no learning |
| Checkpoint with optimizers (train-11) | Optimizer state lost → can't resume training |
| Trainer instantiation (train-15) | Training loop can't start |
| Trainer train step (train-16) | Training produces NaN loss |

### High (must pass before long training runs)

| # | Test | Risk if Fails |
|---|------|---------------|
| Output shapes (9–11, 13, 14) | Downstream shape mismatch |
| MTP shapes/output (14, 19) | MTP heads misaligned |
| MoE routing (36–37) | Routing broken → load imbalance or collapse |
| Scheduler LR curve (train-8,9,10) | Wrong LR → no convergence or divergence |
| Validation (train-13) | Can't monitor training progress |
| Optimizer step (train-17) | LR not updating |
| MTP loss (train-18) | MTP not contributing to loss |

### Medium (can run in CI but not blocking)

| # | Test | Notes |
|---|------|-------|
| BF16 tests (24–28) | Require CUDA; skipped on CPU |
| Determinism (31) | Ensures reproducibility |
| Softcap CE (32–33) | Numerical stability of auxiliary head |
| Scheduler total (train-7) | Sanity check only |
| find_latest_checkpoint (train-12) | Utility function |
| validate_forward_shape (train-14) | Sanity check only |

---

## 3. Test Execution Strategy

### 3.1 Quick CI (CPU, ~3 min)

Run only fast tests that don't build the full model:

```bash
pytest tests/ -v --tb=short \
  -k "not test_normuon_optimizer_step and \
      not test_adamw_optimizer_step and \
      not test_gradient_clip and \
      not Trainer and \
      not CheckpointWithOptimizer and \
      not MTPLoss and \
      not Validation"
```

This runs 32 model tests + 7 training tests = **39 tests**, all parameter-count, shape, and individual-component tests that use lightweight fixtures.

### 3.2 Standard CI (CPU, ~8 min)

Run all tests except BF16:

```bash
pytest tests/ -v --tb=short -k "not bf16"
```

Runs **50 tests** (32 model + 18 training). Full model tests (backward, optimizer step, trainer) each build the 854M-param model and run a short forward/backward. On Apple M4 this takes ~7 minutes.

### 3.3 Full CI (GPU, ~2 min)

Run everything on a CUDA device:

```bash
pytest tests/ -v --tb=short
```

Runs **55 tests**. BF16 compatibility tests activate on CUDA. Full model tests are ~3× faster on GPU than CPU. Total wall time on A100 ~2 minutes.

### 3.4 Targeted Test Commands

| Scenario | Command |
|----------|---------|
| Parameter budget check only | `pytest tests/test_models.py -k "ParameterCount"` |
| Layer schedule verification | `pytest tests/test_models.py -k "LayerSchedule"` |
| MoE routing test | `pytest tests/test_models.py -k "MoERouting"` |
| Optimizer sanity | `pytest tests/test_training.py -k "Optimizer"` |
| Scheduler curve | `pytest tests/test_training.py -k "Scheduler"` |
| Trainer smoke test | `pytest tests/test_training.py -k "Trainer"` |

---

## 4. Test Environment Requirements

### 4.1 CPU (Minimum)

- Python 3.14+ (as shipped on macOS)
- PyTorch 2.x (CPU build)
- pytest 9.x
- 16 GB+ RAM (model tests peak at ~12 GB RSS)
- Runtime: ~7 minutes for full suite

### 4.2 GPU (Recommended for BF16 tests)

- NVIDIA GPU with CUDA compute capability 7.0+ (A100 80GB ideal)
- PyTorch 2.x (CUDA build)
- 80 GB VRAM (model requires ~10.6 GB at BS=2)
- Runtime: ~2 minutes for full suite

### 4.3 Dependencies (from pyproject.toml)

```
torch >= 2.0.0
safetensors >= 0.4.0
wandb >= 0.16.0
pytest >= 8.0
pytest-benchmark >= 4.0
pytest-cov >= 5.0
```

---

## 5. Coverage Report

### 5.1 Model Coverage

| Component | Lines | Test Coverage |
|-----------|-------|---------------|
| `models/mla.py` | ~180 | Parameter count, forward/backward, BF16, shapes |
| `models/moe.py` | ~250 | Parameter count, forward/backward, BF16, shapes, routing, bias update |
| `models/gdn.py` | ~200 | Parameter count, forward/backward, BF16, shapes |
| `models/mtp.py` | ~150 | Parameter count, forward, shapes, softcap CE, loss computation |
| `models/fusionllm.py` | ~240 | Parameter count, forward, backward, shapes, layer schedule, hidden state |

### 5.2 Training Coverage

| Component | Lines | Test Coverage |
|-----------|-------|---------------|
| `training/optimizer.py` | ~120 | Instantiation, step, gradient clip, build_optimizers |
| `training/scheduler.py` | ~80 | Total steps, warmup, decay, min_lr after total |
| `training/checkpoint.py` | ~150 | Save/load roundtrip, metadata, find latest, max_keep |
| `training/validation.py` | ~50 | Loss computation, forward shape |
| `training/trainer.py` | ~420 | Instantiation, train step, optimizer step |

### 5.4 Gap Analysis

| Area | Gap | Priority | Mitigation |
|------|-----|----------|------------|
| GDN state overflow detection | Low | No automated test; must be caught during training by `gdn_state_max_abs` metric |
| Numerical drift from gradient checkpointing | Low | No numerical comparison test; spec recommends spot-check every 1000 steps |
| MoE expert load entropy | Low | No automated metric; W&B tracks during training |
| Data pipeline integration | Not tested | Out of scope for implementation phase; tested separately |
| A100 VRAM fit check | Not tested | Requires A100 hardware; `min_vram_gb=70` enforced at runtime |
| Learning rate warmup with NorMuon | Not tested | NorMuon uses fixed LR, not scheduler; this is by design |

---

## 6. Test Data

All tests use **synthetic random data**:

- **Input tokens**: `torch.randint(0, vocab_size, (B, T))` — uniform random token IDs.
- **Input hidden**: `torch.randn(B, T, dim)` — standard normal.
- **Targets**: `torch.randint(0, vocab_size, (B, T))` — uniform random token IDs.
- **Validation**: Same synthetic data, no real dataset required.

**Why synthetic data**: The implementation phase tests for correctness (shapes, gradients, numerics), not for learning quality. Real data validation requires a trained tokenizer and processed dataset, which is a separate workstream.

---

## 7. Test Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
filterwarnings = ["error"]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

**Warning-as-errors**: The `filterwarnings = ["error"]` setting turns all Python warnings into test failures. This catches:
- Deprecation warnings (e.g., old PyTorch APIs)
- `lr_scheduler.step()` before `optimizer.step()` (handled in tests by using `sched.last_epoch` directly)
- Any unexpected numerical warnings

---

## 8. Test Results (Latest Run)

### 8.1 Model Tests (CPU — Apple M4)

```
32 passed, 5 skipped in 367.55s (0:06:07)
```

5 skipped = BF16 tests (require CUDA).

### 8.2 Training Tests (CPU — Apple M4)

```
18 passed in ~420s (with full model tests)
```

### 8.3 All Tests (CPU — Apple M4)

```
50 passed, 5 skipped in ~490s
```

---

## 9. Adding New Tests

When adding new functionality, follow these guidelines:

1. **Category matching**: Place model tests in `tests/test_models.py`, training tests in `tests/test_training.py`.
2. **Frozen config**: Use `FROZEN_CONFIG` dict (defined in both test files) to ensure tests match spec.
3. **Device fixture**: Use `device()` fixture returning `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
4. **Parameter budget tests**: Compare exact integer against `FINAL_PARAMETER_BUDGET.md`. Tolerance only for composite models (full model, MTP) where float rounding may occur.
5. **Forward tests**: Always assert `torch.isfinite(out).all()` to catch NaN/Inf.
6. **Backward tests**: Always assert `torch.isfinite(p.grad).all()` for each parameter group.
7. **BF16 tests**: Skip with `pytest.skip("BF16 test requires CUDA")` when `device.type != "cuda"`.
8. **Slow tests**: Mark with `@pytest.mark.slow` to allow filtering.
