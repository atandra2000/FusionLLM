# Codebase Simplification Audit

**Date**: 2025-01-07  
**Scope**: Maintainability and code deduplication refactor  
**Constraint**: All training behavior preserved, all safety checks retained

---

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total lines (modified files) | 3,602 | 3,473 | -129 (3.6%) |
| New utility file | — | 64 | +64 |
| Net reduction | — | — | -65 lines (1.8%) |
| Duplicate implementations removed | — | — | 5 |
| Functions consolidated | — | — | 8 |

---

## Changes by Category

### 1. Tensor Validation Consolidation

**New file**: `utils/tensor_checks.py` (64 lines)

**Rationale**: NaN/Inf validation was duplicated in 4 locations with identical logic:
- `training/pretrain.py:837-846` (gradient check in `_optimizer_step`)
- `training/pretrain.py:915-922` (loss check in `train_step`)
- `training/numerical_health.py:91-99` (loss check in `update_loss`)
- `training/numerical_health.py:174-183` (gradient check in `update_gradients`)

**Solution**: Created shared utilities:
- `validate_scalar(value, name, step)` — for loss values
- `validate_tensor(tensor, name, step)` — for tensors
- `validate_gradients(model, step)` — for all model gradients
- `validate_loss(loss, step)` — for loss tensors

**Risk**: LOW — Same error messages, same behavior  
**Benefit**: Single implementation, consistent error formatting

**Lines affected**:
- `training/pretrain.py`: -14 lines (inline checks → 2 function calls)
- `training/numerical_health.py`: -14 lines (inline checks → 2 function calls)

---

### 2. Legacy TrainingConfig Removal

**File**: `training/pretrain.py`

**Rationale**: `TrainingConfig` (93 lines) was a flat dataclass duplicating `ConfigBundle`. 
It was:
- Defined in `pretrain.py` but never instantiated
- Exported in `training/__init__.py` but never imported by other modules
- Documented as "legacy" with recommendation to migrate to `ConfigBundle`

**Solution**: Removed `TrainingConfig` class and its export from `__init__.py`.

**Risk**: LOW — No code used this class  
**Benefit**: -93 lines, reduced confusion between two config classes

**Documentation**: Already documented in `docs/07_RISKS_AND_TECHNICAL_DEBT.md` as tech debt

---

### 3. MoE Expert Forward Consolidation

**File**: `models/moe.py`

**Rationale**: Expert computation logic was duplicated in 3 locations:
1. `forward()` scatter-gather path (lines 537-551)
2. `_all_to_all_dispatch()` fallback path (lines 720-733)
3. Shared expert computation (lines 567-592)

All contained identical activation logic:
```python
h1 = F.linear(x_group, w1)
if self.activation == "swiglu":
    h3 = F.linear(x_group, w3)
    h = F.silu(h1) * h3
else:
    h = torch.relu(h1).square()
expert_out = F.linear(h, w2)
```

**Solution**: Extracted helper methods:
- `_expert_forward_single(x, w1, w2, w3)` — single expert forward
- `_compute_shared_experts(flat)` — all shared experts summed

**Risk**: LOW — Same computation, just extracted  
**Benefit**: -37 lines, single implementation of expert forward

---

### 4. Checkpoint Metadata Consolidation

**File**: `utils/checkpoint.py`

**Rationale**: Metadata building and best-tracking logic was duplicated in 5 save paths:
- `save()`
- `save_state_dict()`
- `_execute_save_gathered()`
- `_save_sharded_state()`
- `_save_sync_state()`
- `_execute_save_fsdp2_dcp()`

Each contained:
```python
meta = {"step": step, "best_val_loss": self.best_val_loss}
if val_loss is not None:
    meta["val_loss"] = val_loss
if extra_meta:
    meta.update({k: v for k, v in extra_meta.items() if k != "step"})
```

And the best-tracking:
```python
with self._best_lock:
    if val_loss is not None and val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        self._update_best(...)
```

**Solution**: Extracted helpers:
- `_build_meta(step, val_loss, extra_meta)` — metadata construction
- `_maybe_update_best(state, ema_state, step, val_loss)` — best tracking

**Risk**: LOW — Thread-safe, same behavior  
**Benefit**: -30 lines, consistent metadata handling, fixed a minor bug (spurious lock in `_execute_save_gathered`)

---

### 5. _all_to_all_dispatch Consolidation

**File**: `models/moe.py`

**Rationale**: `_all_to_all_dispatch()` was a stub that duplicated the scatter-gather path from `forward()`, including the inline expert computation.

**Solution**: Updated to use `_expert_forward_single()` helper, removing duplicate activation logic.

**Risk**: LOW — Same fallback behavior  
**Benefit**: Consistent with forward() path

---

## What Was NOT Changed

Per hard constraints, the following were preserved:

- ✅ NaN/Inf protection (consolidated, not removed)
- ✅ Gradient clipping
- ✅ Router z-loss
- ✅ FSDP support
- ✅ MoE support
- ✅ MLA support
- ✅ MTP support
- ✅ Checkpoint recovery
- ✅ Optimizer state recovery
- ✅ Curriculum recovery
- ✅ All existing tests (unchanged)

---

## Files Modified

| File | Lines Before | Lines After | Change |
|------|-------------|-------------|--------|
| `training/pretrain.py` | 1,306 | 1,195 | -111 |
| `training/numerical_health.py` | 370 | 356 | -14 |
| `utils/checkpoint.py` | 1,090 | 1,060 | -30 |
| `models/moe.py` | 822 | 785 | -37 |
| `utils/tensor_checks.py` | — | 64 | +64 (new) |
| `training/__init__.py` | 14 | 13 | -1 |
| **Total** | **3,602** | **3,473** | **-129** |

---

## Risk Assessment

| Change | Risk Level | Mitigation |
|--------|-----------|------------|
| Tensor validation consolidation | LOW | Same error messages, same behavior |
| TrainingConfig removal | LOW | Class was never used |
| MoE expert consolidation | LOW | Extracted, not rewritten |
| Checkpoint metadata consolidation | LOW | Helper functions, same logic |
| _all_to_all_dispatch consolidation | LOW | Uses same helper as forward() |

**Overall Risk**: LOW — All changes are structural (extract method / remove dead code) with no behavioral changes.

---

## Verification

- All existing tests should continue to pass
- No training behavior changes
- No configuration changes required
- No API changes (all public interfaces preserved)
