# Training-Readiness Audit

**Date**: 2026-06-06  
**Auditor**: Training Infrastructure Auditor  
**Scope**: Complete training-readiness audit across 20 areas  
**Goal**: Determine whether the repository is ready for a multi-billion-token training run

---

## Executive Summary

The codebase has strong foundations — clean MLA, well-structured FSDP2, async checkpointing, and dual logging. However, several critical gaps remain that could cause silent training failures, lost optimizer state on restart, or unstable routing. The most impactful issues are:

1. **NorMuon/Muon optimizer state is not saved/loaded** — restart resets optimizer momentum
2. **Router z-loss is not applied** — router instability unregulated
3. **No NaN/Inf detection** — silent training corruption possible
4. **EMA is dead code** — infrastructure exists but no implementation
5. **Grad norm not logged** — invisible gradient health

**Overall Readiness: 7/10** — Functional but not production-safe for multi-billion-token runs.

---

## 1. Training Loop

**File**: `training/pretrain.py:856-910`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| No NaN/Inf check on loss before backward | 🔴 HIGH | Silent training corruption — NaN gradients propagate undetected | Medium | Add `torch.isnan(loss).any()` check before `scaler.scale(loss).backward()` |
| GradScaler instantiated with `enabled=False` but still called | 🟡 LOW | Unnecessary overhead (negligible) | High | Use `contextlib.nullcontext()` when scaler disabled |
| `_opt_steps` counter incremented even when no actual optimizer step occurs (gradient accumulation boundary bug) | 🟡 MEDIUM | Bias updates fire at wrong frequency | Low | Already correct — `is_opt_step` gates the increment |
| `train_step` uses `micro_step` (global_step) for gradient accumulation but `is_opt_step` checks `(micro_step + 1) % grad_accum == 0` — this means the FIRST step (micro_step=0) is NOT an opt step if grad_accum > 1 | 🟡 LOW | First micro-batch gradients are accumulated correctly, but the naming is confusing | Low | Rename parameter to `global_step` for clarity |

---

## 2. Checkpointing

**File**: `utils/checkpoint.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| `keep_last_n` not called in safetensors save path | 🟡 MEDIUM | Disk fills up over long runs | High | Add `keep_last_n` call after safetensors save |
| `val_loss` and `ema_state` never passed from `save_checkpoint` in pretrain.py | 🟡 MEDIUM | Best checkpoint tracking unused | High | Pass `val_loss` from eval results; implement EMA |
| Atomic writes via temp+rename — good | ✅ OK | — | — | — |
| DCP backend support — good | ✅ OK | — | — | — |
| `best_val_loss` persisted across restarts — good | ✅ OK | — | — | — |
| Shared tensor deduplication for tied embeddings — good | ✅ OK | — | — | — |

---

## 3. Resume/Restart Logic

**File**: `training/pretrain.py:935-956, 1025-1031`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| NorMuon/Muon optimizer state NOT saved/loaded | 🔴 CRITICAL | Optimizer momentum reset on restart — training effectively restarts from scratch with random gradients | High | Save/load NorMuon state in `save_checkpoint`/`load_checkpoint` |
| Curriculum state NOT saved | 🟡 MEDIUM | Curriculum re-advances or misses switch on restart | Medium | Save `curriculum._advanced` flag in checkpoint metadata |
| Health monitor state NOT saved | 🟢 LOW | Spike count and loss history reset on restart | Low | Save health_monitor stats in checkpoint metadata |
| `_opt_steps` IS restored — good | ✅ OK | — | — | — |
| Scheduler state IS restored — good | ✅ OK | — | — | — |
| `_find_latest_checkpoint()` auto-discovers latest — good | ✅ OK | — | — | — |

---

## 4. EMA Support

**File**: `utils/checkpoint.py` (infrastructure), `training/pretrain.py` (missing)

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| EMA infrastructure exists (ema_state, best_ema.safetensors) but NO actual EMA implementation | 🟡 MEDIUM | Cannot use EMA weights for eval or deployment | High | Implement `torch.optim.swa_utils.AveragedModel` or custom EMA |
| `ema_state` parameter never passed from pretrain.py | 🟡 MEDIUM | EMA never saved | High | Wire EMA into training loop |
| EMA could improve model quality by 1-3% on standard benchmarks | 🟡 MEDIUM | Lost quality improvement | Medium | Implement EMA with configurable decay |

---

## 5. Mixed Precision

**File**: `training/pretrain.py:668-678`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| BF16 autocast — correct | ✅ OK | — | — | — |
| GradScaler `enabled=False` — correct for BF16 | ✅ OK | — | — | — |
| FSDP2 reduce_dtype=FP32 — correct | ✅ OK | — | — | — |
| TF32 enablement in config but never applied | 🟢 LOW | TF32 matmuls not enabled | High | Add `torch.backends.cuda.matmul.allow_tf32 = True` in trainer init |
| cuDNN benchmark in config but never applied | 🟢 LOW | cuDNN auto-tuner not enabled | High | Add `torch.backends.cudnn.benchmark = True` in trainer init |

---

## 6. Gradient Clipping

**File**: `training/pretrain.py:838`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| `clip_grad_norm_` applied to ALL parameters — correct but could be more selective | 🟢 LOW | Minor — clipping non-matrix params is fine | Low | No change needed |
| Gradient norm NOT logged to W&B/MLflow | 🟡 MEDIUM | Invisible gradient health — cannot detect training instability | High | Pass `grad_norm` to `logger.log()` |
| Clipping order (unscale → clip → step) — correct | ✅ OK | — | — | — |
| No gradient norm EMA or history tracking | 🟡 MEDIUM | Cannot detect gradual gradient explosion | Medium | Add gradient norm to health monitor and logging |

---

## 7. NaN Handling

**File**: `training/pretrain.py`, `training/numerical_health.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| No NaN check on loss before backward | 🔴 HIGH | NaN gradients propagate silently — training produces garbage | Medium | Add `if torch.isnan(loss): raise RuntimeError("NaN loss")` |
| No NaN check on gradients before optimizer step | 🔴 HIGH | NaN gradients corrupt optimizer state | Medium | Add NaN check in `_optimizer_step` |
| Health monitor checks z-score but NOT NaN/Inf | 🟡 MEDIUM | Spike detection misses NaN/Inf | Medium | Add `torch.isnan`/`torch.isinf` checks in `update_loss` |
| No `torch.autograd.set_detect_anomaly(True)` | 🟢 LOW | NaN root cause harder to trace | Low | Add as opt-in via env var |

---

## 8. Inf Handling

**File**: `training/pretrain.py`, `models/moe.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| No Inf check on loss or gradients | 🔴 HIGH | Same as NaN — silent corruption | Medium | Add Inf checks alongside NaN checks |
| Router logits clamped to ±30 — good | ✅ OK | Prevents extreme router values | — | — |
| No Inf check on activations | 🟡 MEDIUM | Inf activations could propagate | Low | Enable activation monitoring in health check |

---

## 9. Loss Scaling

**File**: `training/pretrain.py:678, 895`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| GradScaler `enabled=False` — correct for BF16 | ✅ OK | — | — | — |
| No dynamic loss scaling needed for BF16 | ✅ OK | — | — | — |
| `scaler.scale(loss).backward()` is a no-op when disabled — fine | ✅ OK | — | — | — |

---

## 10. Optimizer State Recovery

**File**: `training/pretrain.py:935-956, utils/checkpoint.py:664-686`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| NorMuon/Muon state NOT saved/loaded | 🔴 CRITICAL | Optimizer momentum, exp_avg, exp_avg_sq all lost on restart — training regresses | High | Add NorMuon state to checkpoint save/load |
| AdamW state IS saved/loaded — good | ✅ OK | — | — | — |
| `_opt_steps` IS restored — good | ✅ OK | — | — | — |
| Scheduler state IS restored — good | ✅ OK | — | — | — |
| CautiousAdamW inherits from AdamW — state saved via AdamW — good | ✅ OK | — | — | — |

---

## 11. Data Pipeline

**File**: `data/async_loader.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Async worker errors pushed to queue but trainer doesn't check | 🟡 MEDIUM | Silent data pipeline failure — trainer hangs or uses garbage data | Medium | Add error checking in `__next__` |
| No shard validation on load | 🟢 LOW | Corrupted shards cause silent issues | Low | Add header validation on shard open |
| `_iter_sync` doesn't shuffle across shards properly | 🟢 LOW | Sync mode (smoke tests) has poor sharding | Low | Add cross-shard shuffling in sync path |
| Pinned memory buffer reused — correct | ✅ OK | — | — | — |
| Deterministic per-rank offsets — correct | ✅ OK | — | — | — |
| Curriculum hot-swap support — good | ✅ OK | — | — | — |

---

## 12. Curriculum Transitions

**File**: `data/curriculum.py`, `training/pretrain.py:1043-1051`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Curriculum state NOT saved in checkpoints | 🟡 MEDIUM | Restart re-advances or misses stage switch | Medium | Save `_advanced` flag in checkpoint metadata |
| `advance()` only triggers once — correct | ✅ OK | — | — | — |
| Vose alias sampling — correct | ✅ OK | — | — | — |
| Hot-swap calls `loader.stop()` + `loader.start()` — correct | ✅ OK | — | — | — |

---

## 13. Validation Loop

**File**: `training/pretrain.py:742-810, eval/eval_core.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Validation DISABLED by default (`eval_enabled: false`) | 🟡 MEDIUM | No quality signal during training | High | Enable by default with synthetic data |
| Model not restored to train mode if eval exception occurs | 🟡 MEDIUM | Model stuck in eval mode after failed eval | Low | Add try/finally to restore train mode |
| `run_perplexity` doesn't use `torch.no_grad()` — relies on `@torch.no_grad()` decorator on function | ✅ OK | — | — | — |
| Synthetic loader used for eval — fine for smoke tests | ✅ OK | — | — | — |
| lm-eval-harness integration — good | ✅ OK | — | — | — |

---

## 14. FSDP2 Integration

**File**: `utils/distributed.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Per-TransformerBlock auto-wrap — correct | ✅ OK | — | — | — |
| FULL_SHARD strategy — correct | ✅ OK | — | — | — |
| Backward prefetch on, forward prefetch off — correct | ✅ OK | — | — | — |
| Per-layer reshard tuning — good | ✅ OK | — | — | — |
| Tied parameters (embed/head) not specially handled | 🟢 LOW | Could cause issues with FSDP2 parameter sharding | Low | Verify tied params work with FSDP2 |
| No memory profiling integration | 🟡 MEDIUM | Cannot track memory usage during training | Medium | Integrate memory profiler into training loop |

---

## 15. Memory Leaks

**File**: Various

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| MoE `_last_weights`/`_last_indices` stored but overwritten each step — no leak | ✅ OK | — | — | — |
| Health monitor deques have `maxlen` — no leak | ✅ OK | — | — | — |
| `ActivationMonitor` stores detached activations without auto-cleanup | 🟡 MEDIUM | Memory leak if activated | Low | Add auto-cleanup or warning |
| Async loader pinned buffer reused — no leak | ✅ OK | — | — | — |
| No `torch.cuda.empty_cache()` call in training loop | 🟢 LOW | Fragmentation over long runs | Medium | Add optional cache clearing every N steps |

---

## 16. Autograd Graph Retention

**File**: `models/transformer.py`, `models/moe.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| MLA KV cache: `use_cache=False` during training — correct | ✅ OK | — | — | — |
| MoE buffers: fresh allocation per forward — correct | ✅ OK | — | — | — |
| `use_reentrant=False` in checkpoint — correct | ✅ OK | — | — | — |
| MoE routing weights `.detach()` — prevents gradient flow through routing | 🟡 MEDIUM | Routing decisions don't receive gradients (intentional for AuxLossFree) | High | No change — this is by design |
| `_y_routed_buf` removed, fresh allocation — correct | ✅ OK | — | — | — |

---

## 17. Expert Collapse Detection

**File**: `models/moe.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| No explicit expert collapse detection | 🟡 MEDIUM | All tokens could route to same expert — bias update corrects slowly | Medium | Add monitoring: alert when any expert gets >2× average load |
| `update_bias()` adjusts routing bias — correct | ✅ OK | — | — | — |
| `expert_dropout_prob` during warmup — helps prevent early collapse | ✅ OK | — | — | — |
| Routing histograms logged every 200 steps — manual monitoring only | 🟢 LOW | Delayed detection | Low | Add automated collapse alert |
| No routing entropy tracking over time | 🟡 MEDIUM | Cannot detect gradual routing collapse | Medium | Log routing entropy to W&B |

---

## 18. Router Stability

**File**: `models/moe.py`, `training/pretrain.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Router z-loss NOT applied in training loop | 🔴 HIGH | Router logits can grow unbounded — training instability | Medium | Add `gate.get_z_loss()` to total loss |
| Router logits clamped to ±30 — good | ✅ OK | Prevents extreme values | — | — |
| `AuxLossFreeGate.get_z_loss()` exists but unused | 🟡 MEDIUM | Z-loss regularization available but not applied | High | Wire z-loss into loss computation |
| `balance_loss` uses simple load balance — correct | ✅ OK | — | — | — |
| No router temperature scheduling | 🟢 LOW | Could improve routing quality | Low | Add optional temperature annealing |

---

## 19. Logging

**File**: `utils/logging.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Gradient norm NOT logged to W&B/MLflow | 🟡 MEDIUM | Invisible gradient health | High | Pass `grad_norm` to `logger.log()` |
| NorMuon/Muon LR NOT logged — only AdamW LR | 🟡 MEDIUM | Cannot track primary optimizer LR | High | Log NorMuon/Muon LR separately |
| W&B + MLflow dual logging — good | ✅ OK | — | — | — |
| Async log submissions — good | ✅ OK | — | — | — |
| GPU memory stats logged — good | ✅ OK | — | — | — |
| MoE routing histograms logged — good | ✅ OK | — | — | — |
| Health monitor stats NOT logged to W&B | 🟡 MEDIUM | Spike count, loss EMA not visible | Medium | Log health monitor stats |

---

## 20. Monitoring

**File**: `training/numerical_health.py`

| Issue | Severity | Impact | Probability | Proposed Fix |
|-------|----------|--------|-------------|--------------|
| Loss spike detection — good | ✅ OK | — | — | — |
| Gradient anomaly detection — good | ✅ OK | — | — | — |
| Emergency checkpoint on spike — good | ✅ OK | — | — | — |
| Activation monitoring DISABLED by default | 🟡 MEDIUM | NaN/Inf in activations undetected | Low | Enable with lightweight hooks |
| No MoE routing entropy monitoring | 🟡 MEDIUM | Routing instability undetected | Medium | Add entropy tracking |
| No expert utilization variance tracking | 🟡 MEDIUM | Expert collapse undetected | Medium | Add utilization stats |
| Health monitor stats not logged to W&B | 🟡 MEDIUM | Stats invisible | Medium | Wire to logger |
| No GPU memory tracking in health monitor | 🟢 LOW | Memory leaks undetected | Medium | Add memory stats |

---

## Priority Ranking (Expected Impact on Successful Training)

| Rank | Issue | Severity | Impact | Fix Effort |
|------|-------|----------|--------|------------|
| 1 | NorMuon/Muon optimizer state not saved/loaded | 🔴 CRITICAL | Training restart loses all optimizer momentum — effectively restarts from scratch | Small (2-3 hours) |
| 2 | Router z-loss not applied | 🔴 HIGH | Router instability — can cause training collapse | Small (1 hour) |
| 3 | No NaN/Inf detection on loss/gradients | 🔴 HIGH | Silent training corruption | Small (1-2 hours) |
| 4 | Grad norm not logged | 🟡 MEDIUM | Invisible gradient health — cannot detect instability | Small (30 min) |
| 5 | NorMuon/Muon LR not logged | 🟡 MEDIUM | Cannot track primary optimizer | Small (30 min) |
| 6 | `keep_last_n` missing in safetensors save path | 🟡 MEDIUM | Disk fills up | Small (15 min) |
| 7 | Curriculum state not saved | 🟡 MEDIUM | Restart misses stage switch | Small (1 hour) |
| 8 | EMA not implemented | 🟡 MEDIUM | Lost 1-3% quality improvement | Medium (4-6 hours) |
| 9 | Validation disabled by default | 🟡 MEDIUM | No quality signal during training | Small (15 min) |
| 10 | Async worker errors not checked | 🟡 MEDIUM | Silent data pipeline failure | Small (30 min) |
| 11 | TF32/cuDNN not enabled | 🟢 LOW | Slower training | Small (5 min) |
| 12 | Memory profiler not integrated | 🟡 MEDIUM | Cannot track memory | Small (1 hour) |
| 13 | Health monitor stats not logged | 🟡 MEDIUM | Stats invisible | Small (30 min) |
| 14 | Routing entropy not tracked | 🟡 MEDIUM | Routing instability undetected | Small (1 hour) |
| 15 | Expert collapse detection missing | 🟡 MEDIUM | Collapse detected late | Medium (2-3 hours) |

---

## Readiness Checklist

| Area | Status | Notes |
|------|--------|-------|
| Training loop | ⚠️ READY WITH FIXES | Needs NaN/Inf checks |
| Checkpointing | ⚠️ READY WITH FIXES | Needs keep_last_n fix, val_loss wiring |
| Resume/restart | 🔴 NOT READY | NorMuon state not saved/loaded |
| EMA | 🟡 PARTIAL | Infrastructure only, no implementation |
| Mixed precision | ✅ READY | BF16 correct |
| Gradient clipping | ⚠️ READY WITH FIXES | Needs grad norm logging |
| NaN handling | 🔴 NOT READY | No detection |
| Inf handling | 🔴 NOT READY | No detection |
| Loss scaling | ✅ READY | Correct for BF16 |
| Optimizer state recovery | 🔴 NOT READY | NorMuon state lost |
| Data pipeline | ⚠️ READY WITH FIXES | Needs error checking |
| Curriculum transitions | ⚠️ READY WITH FIXES | Needs state saving |
| Validation loop | ⚠️ READY WITH FIXES | Needs enable by default |
| FSDP2 integration | ✅ READY | Well-configured |
| Memory leaks | ✅ READY | No leaks detected |
| Autograd graph retention | ✅ READY | Phase 3 fixes correct |
| Expert collapse detection | ⚠️ PARTIAL | Bias update exists, monitoring missing |
| Router stability | 🔴 NOT READY | Z-loss not applied |
| Logging | ⚠️ READY WITH FIXES | Needs grad norm + LR logging |
| Monitoring | ⚠️ PARTIAL | Basic monitoring exists, advanced missing |

---

## Recommended Fix Order

### Immediate (Before Training)
1. Save/load NorMuon optimizer state
2. Add NaN/Inf checks on loss and gradients
3. Apply router z-loss
4. Fix `keep_last_n` in safetensors path

### Short-term (First 1000 Steps)
5. Log gradient norm to W&B
6. Log NorMuon/Muon LR
7. Enable validation by default
8. Add async worker error checking

### Medium-term (Before Large-Scale Run)
9. Implement EMA
10. Add curriculum state saving
11. Enable TF32/cuDNN
12. Integrate memory profiler

### Long-term (During Training)
13. Add expert collapse detection
14. Add routing entropy monitoring
15. Add activation monitoring

---

## Conclusion

The repository is **functional but not production-safe** for multi-billion-token runs. The critical blocker is the NorMuon optimizer state not being saved/loaded — this means every restart effectively resets training. The router z-loss and NaN/Inf detection are also critical for training stability.

**Estimated fix effort for critical issues: 4-6 hours**  
**Estimated fix effort for all issues: 2-3 days**

After the critical fixes, the codebase should be ready for a multi-billion-token training run on 8×A100.
