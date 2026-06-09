# Phase 3 — Architectural Modularization Refactor Plan

## Executive Summary

This plan details the architectural modularization of the FusionLLM codebase to improve maintainability, reduce cognitive complexity, and separate responsibilities. The goal is to transform a large research-style codebase into a production ML framework that another engineer can understand in one day instead of one week.

**Current State:**
- `training/pretrain.py`: 1,195 lines — monolithic training loop
- `utils/checkpoint.py`: 1,060 lines — mixed concerns
- `models/moe.py`: 785 lines — multiple responsibilities
- Total: ~7,759 lines across all modules

**Target State:**
- `training/pretrain.py`: <500 lines (thin entrypoint)
- `utils/checkpoint/`: 5-6 focused modules
- `models/moe/`: 5-6 focused modules
- Clear dependency flow: configs → models → training → entrypoints

---

## Priority 1: Split training/pretrain.py

### Current Structure Analysis

`pretrain.py` contains 1,195 lines with 17 classes/functions:

| Component | Lines | Responsibility |
|-----------|-------|----------------|
| WarmupCosineDecayScheduler | 27 | LR scheduling |
| ConfigBundle + sub-configs | 105 | Configuration dataclasses |
| PretrainDataset | 27 | Dataset loading |
| Muon optimizer | 56 | Newton-Schulz orthogonalization |
| CautiousAdamW | 26 | Sign-masked weight decay |
| _build_optimizers | 79 | Optimizer factory |
| Pretrainer class | 613 | **Everything else** |

### Proposed Structure

```
training/
├── pretrain.py          # Thin entrypoint (~400 lines)
├── trainer.py           # Pretrainer class orchestration (~300 lines)
├── optimization.py      # Muon, CautiousAdamW, _build_optimizers (~200 lines)
├── validation.py        # Evaluation logic (~150 lines)
├── checkpointing.py     # Checkpoint save/load orchestration (~100 lines)
├── curriculum_manager.py # Curriculum transitions (~80 lines)
├── monitoring.py        # Health monitoring, metrics collection (~100 lines)
├── train_step.py        # train_step, _optimizer_step (~150 lines)
├── configs.py           # ConfigBundle and sub-configs (~120 lines)
├── dataset.py           # PretrainDataset (~40 lines)
├── loss.py              # (unchanged)
├── numerical_health.py  # (unchanged)
├── normuon.py           # (unchanged)
├── schedules.py         # (unchanged)
└── wsd.py               # (unchanged)
```

### Detailed Migration Plan

#### 1.1 Create `training/configs.py`
**Move:** All dataclass definitions (lines 98-200)
- `DataConfig`, `OptimConfig`, `ScheduleConfig`, `EvalConfig`, `CheckpointConfig`, `LoggingConfig`, `ConfigBundle`
- **Dependencies:** `dataclasses`, `typing`
- **Exports:** All config classes

#### 1.2 Create `training/optimization.py`
**Move:** Optimizer classes and builder (lines 238-441)
- `Muon` class
- `CautiousAdamW` class
- `_zeropower_via_newtonschulz5()` helper
- `_cautious_mask()` helper
- `_build_optimizers()` factory
- **Dependencies:** `torch`, `torch.optim`, `training.normuon`
- **Exports:** `Muon`, `CautiousAdamW`, `build_optimizers`

#### 1.3 Create `training/dataset.py`
**Move:** Dataset class (lines 206-232)
- `PretrainDataset`
- **Dependencies:** `torch`, `os`
- **Exports:** `PretrainDataset`

#### 1.4 Create `training/train_step.py`
**Move:** Training step logic from Pretrainer (lines 769-838, 738-767)
- `train_step()` method → `train_step()` function
- `_optimizer_step()` method → `optimizer_step()` function
- **Dependencies:** `torch`, `utils.tensor_checks`, `utils.distributed`
- **Exports:** `train_step`, `optimizer_step`

#### 1.5 Create `training/validation.py`
**Move:** Evaluation logic from Pretrainer (lines 650-718)
- `_maybe_eval()` method → `maybe_eval()` function
- **Dependencies:** `eval.eval_core`, `eval.run_lm_eval`, `utils.logging`
- **Exports:** `maybe_eval`

#### 1.6 Create `training/checkpointing.py`
**Move:** Checkpoint save/load orchestration (lines 840-931)
- `save_checkpoint()` method → `save_checkpoint()` function
- `load_checkpoint()` method → `load_checkpoint()` function
- `_verify_nor_muon_state()` helper
- `_find_latest_checkpoint()` helper
- **Dependencies:** `utils.checkpoint`, `utils.distributed`
- **Exports:** `save_checkpoint`, `load_checkpoint`, `find_latest_checkpoint`

#### 1.7 Create `training/curriculum_manager.py`
**Move:** Curriculum management (lines 599-607, 1015-1023)
- Curriculum initialization
- Curriculum advance logic
- **Dependencies:** `data.curriculum`, `pathlib`
- **Exports:** `init_curriculum`, `advance_curriculum`

#### 1.8 Create `training/monitoring.py`
**Move:** Monitoring setup and health check integration (lines 614-629, 813-823)
- Health monitor initialization
- Metrics collection
- **Dependencies:** `training.numerical_health`, `utils.logging`
- **Exports:** `init_health_monitor`, `log_metrics`

#### 1.9 Refactor `training/trainer.py`
**New file:** Thin orchestration layer (~300 lines)
```python
class Pretrainer:
    def __init__(self, config: ConfigBundle):
        # Initialize components
        self.model = setup_model(config)
        self.optimizers = build_optimizers(self.model, config)
        self.ckpt_manager = setup_checkpointing(config)
        self.health_monitor = init_health_monitor(config)
        self.curriculum = init_curriculum(config)
        
    def train(self):
        # Orchestrate training loop
        for step in range(max_steps):
            advance_curriculum(self.curriculum, step)
            tokens, targets = next(loader)
            metrics = train_step(self.model, tokens, targets, step)
            if should_eval(step):
                maybe_eval(self.model, step)
            if should_save(step):
                save_checkpoint(self.ckpt_manager, step)
```

#### 1.10 Refactor `training/pretrain.py`
**Final state:** Thin entrypoint (~400 lines)
- ConfigBundle imports (from configs.py)
- `build_config_from_yaml()` function
- `main()` function
- CLI argument parsing

### Migration Sequence

1. Create `configs.py` first (no dependencies)
2. Create `dataset.py` (no dependencies)
3. Create `optimization.py` (depends on normuon)
4. Create `train_step.py` (depends on tensor_checks, distributed)
5. Create `validation.py` (depends on eval module)
6. Create `checkpointing.py` (depends on utils.checkpoint)
7. Create `curriculum_manager.py` (depends on data.curriculum)
8. Create `monitoring.py` (depends on numerical_health)
9. Create `trainer.py` (depends on all above)
10. Refactor `pretrain.py` to use new modules

---

## Priority 2: Split utils/checkpoint.py

### Current Structure Analysis

`checkpoint.py` contains 1,060 lines with mixed concerns:

| Component | Lines | Responsibility |
|-----------|-------|----------------|
| CheckpointManager.__init__ | 32 | Initialization |
| Async worker | 63 | Background thread management |
| Save methods | 200+ | Multiple save paths |
| Load methods | 100+ | Multiple load paths |
| DCP support | 200+ | FSDP2 distributed checkpoint |
| Helpers | 100+ | Atomic writes, metadata |
| Retention | 50+ | keep_last_n, delete |

### Proposed Structure

```
utils/checkpoint/
├── __init__.py      # Re-export CheckpointManager (~20 lines)
├── manager.py       # CheckpointManager orchestrator (~200 lines)
├── metadata.py      # _build_meta, _load_best_val_loss (~80 lines)
├── retention.py     # keep_last_n, delete_checkpoint (~80 lines)
├── fsdp.py          # FSDP2 save/load (~200 lines)
├── dcp.py           # DCP save/load (~200 lines)
├── recovery.py      # load, load_weights, state restoration (~150 lines)
├── atomic.py        # Atomic write helpers (~100 lines)
└── async_worker.py  # Async thread management (~100 lines)
```

### Detailed Migration Plan

#### 2.1 Create `utils/checkpoint/atomic.py`
**Move:** Atomic write helpers (lines 967-1019)
- `_atomic_save_safetensors()`
- `_atomic_save_torch()`
- `_atomic_save_json()`
- `_json_default()` helper
- **Dependencies:** `json`, `os`, `tempfile`, `torch`, `safetensors`
- **Exports:** All atomic save functions

#### 2.2 Create `utils/checkpoint/metadata.py`
**Move:** Metadata handling (lines 896-914, 568-582, 916-928)
- `_build_meta()`
- `_load_best_val_loss()`
- `_maybe_update_best()`
- `_update_best()`
- **Dependencies:** `json`, `logging`
- **Exports:** `build_meta`, `load_best_val_loss`, `maybe_update_best`

#### 2.3 Create `utils/checkpoint/retention.py`
**Move:** Checkpoint retention logic (lines 930-961)
- `keep_last_n()`
- `delete_checkpoint()`
- `list_checkpoints()`
- `latest_step()`
- `_list_steps()`
- `_checkpoint_complete()`
- **Dependencies:** `pathlib`, `logging`
- **Exports:** `keep_last_n`, `delete_checkpoint`, `list_checkpoints`, `latest_step`

#### 2.4 Create `utils/checkpoint/fsdp.py`
**Move:** FSDP2 save/load (lines 705-808)
- `save_fsdp2_dcp()`
- `_execute_save_fsdp2_dcp()`
- `_save_sharded_state()`
- `_save_best_sharded()`
- `_combine_meta()`
- **Dependencies:** `torch.distributed`, `torch.distributed.checkpoint`
- **Exports:** `save_fsdp2_dcp`, `execute_save_fsdp2_dcp`

#### 2.5 Create `utils/checkpoint/dcp.py`
**Move:** DCP-specific logic (lines 837-890)
- `load_fsdp2_dcp()`
- DCP state dict options
- **Dependencies:** `torch.distributed.checkpoint`
- **Exports:** `load_fsdp2_dcp`

#### 2.6 Create `utils/checkpoint/recovery.py`
**Move:** Standard load methods (lines 588-699)
- `load()`
- `load_weights()`
- **Dependencies:** `safetensors`, `torch`
- **Exports:** `load`, `load_weights`

#### 2.7 Create `utils/checkpoint/async_worker.py`
**Move:** Async thread management (lines 105-164)
- `_start_async_worker()`
- `_stop_async_worker()`
- `_async_worker_loop()`
- `_save_sync()`
- `save_async()`
- `delete_async()`
- **Dependencies:** `threading`, `queue`
- **Exports:** `AsyncCheckpointWorker`

#### 2.8 Create `utils/checkpoint/manager.py`
**New file:** Orchestrator (~200 lines)
```python
class CheckpointManager:
    def __init__(self, ...):
        # Initialize sub-components
        self.atomic = AtomicWriter(save_dir)
        self.retention = RetentionManager(save_dir)
        self.metadata = MetadataManager(save_dir)
        self.async_worker = AsyncCheckpointWorker(self) if async_mode else None
        
    def save(self, model, optimizer, step, ...):
        # Delegate to appropriate backend
        state = model.state_dict()
        self.atomic.save_safetensors(state, path)
        meta = self.metadata.build_meta(step, ...)
        self.metadata.maybe_update_best(state, ...)
        if keep_last_n:
            self.retention.keep_last_n(keep_last_n)
```

#### 2.9 Update `utils/checkpoint/__init__.py`
```python
from .manager import CheckpointManager
__all__ = ["CheckpointManager"]
```

---

## Priority 3: Split models/moe.py

### Current Structure Analysis

`moe.py` contains 785 lines with 3 classes and multiple functions:

| Component | Lines | Responsibility |
|-----------|-------|----------------|
| compute_routing_segments | 56 | Sorting/segment logic |
| AuxLossFreeGate | 112 | Routing decision |
| Expert | 57 | Expert FFN |
| DeepSeekMoE | 535 | **Everything else** |

### Proposed Structure

```
models/moe/
├── __init__.py       # Re-export DeepSeekMoE (~20 lines)
├── router.py         # AuxLossFreeGate (~150 lines)
├── experts.py        # Expert class (~80 lines)
├── balancing.py      # Load balancing loss (~100 lines)
├── dispatch.py       # Scatter-gather, all-to-all (~250 lines)
├── monitoring.py     # Routing stats (~80 lines)
├── weight_stacks.py  # Precomputed weight stacks (~100 lines)
└── moe.py            # DeepSeekMoE coordinator (~200 lines)
```

### Detailed Migration Plan

#### 3.1 Create `models/moe/router.py`
**Move:** Gate class (lines 77-188)
- `AuxLossFreeGate` class
- **Dependencies:** `torch`, `torch.nn`, `torch.nn.functional`
- **Exports:** `AuxLossFreeGate`

#### 3.2 Create `models/moe/experts.py`
**Move:** Expert class (lines 191-247)
- `Expert` class
- **Dependencies:** `torch`, `torch.nn`, `torch.nn.functional`, `torch.distributed`
- **Exports:** `Expert`

#### 3.3 Create `models/moe/balancing.py`
**Move:** Load balancing logic (lines 729-750)
- `get_load_balance_loss()`
- `get_z_loss()`
- `_get_weighted_onehot()`
- **Dependencies:** `torch`
- **Exports:** `get_load_balance_loss`, `get_z_loss`

#### 3.4 Create `models/moe/dispatch.py`
**Move:** Dispatch logic (lines 438-723)
- `forward()` scatter-gather path
- `_try_grouped_gemm()`
- `_all_to_all_dispatch()`
- `compute_routing_segments()` (top-level function)
- **Dependencies:** `torch`, `torch.distributed`, `ops.triton.grouped_gemm`
- **Exports:** `dispatch_tokens`, `compute_routing_segments`

#### 3.5 Create `models/moe/monitoring.py`
**Move:** Routing stats (lines 752-767)
- `get_routing_stats()`
- **Dependencies:** `torch`
- **Exports:** `get_routing_stats`

#### 3.6 Create `models/moe/weight_stacks.py`
**Move:** Weight stack management (lines 379-405)
- `_refresh_weight_stacks()`
- **Dependencies:** `torch`
- **Exports:** `refresh_weight_stacks`

#### 3.7 Create `models/moe/moe.py`
**New file:** Coordinator (~200 lines)
```python
class DeepSeekMoE(nn.Module):
    def __init__(self, config, ...):
        self.gate = AuxLossFreeGate(config)
        self.experts = nn.ModuleList([Expert(...) for _ in range(n_local)])
        self.shared_experts = nn.ModuleList([Expert(...) for _ in range(n_shared)])
        
    def forward(self, x):
        weights, indices = self.gate(x)
        y_routed = dispatch_tokens(x, weights, indices, self.experts)
        y_shared = compute_shared_experts(x, self.shared_experts)
        return y_routed + y_shared
```

#### 3.8 Update `models/moe/__init__.py`
```python
from .moe import DeepSeekMoE
__all__ = ["DeepSeekMoE"]
```

---

## Priority 4: Consolidate Training Utilities

### Audit Findings

| Location | Duplicated Function | Consolidation Target |
|----------|-------------------|---------------------|
| `training/pretrain.py:631` | `_log()` | `utils/logging.py` |
| `training/pretrain.py:625` | `_on_spike()` callback | `training/monitoring.py` |
| `utils/logging.py` | `RunsCsvLogger` | `utils/logging.py` (keep) |
| `utils/tensor_checks.py` | `validate_*` functions | Already consolidated ✓ |
| `training/numerical_health.py` | Alert callbacks | `training/monitoring.py` |

### Proposed Consolidations

1. **Move `RunsCsvLogger` to standalone module** if needed for reuse
2. **Create `training/metrics.py`** for metric aggregation patterns
3. **Create `training/logging_utils.py`** for training-specific logging helpers

---

## Priority 5: Dependency Direction Audit

### Current Dependencies (Problematic)

```
training/pretrain.py → utils/checkpoint.py ✓
training/pretrain.py → utils/distributed.py ✓
training/pretrain.py → utils/logging.py ✓
training/pretrain.py → utils/tensor_checks.py ✓
training/pretrain.py → training.normuon ✓
training/pretrain.py → training.schedules ✓
training/pretrain.py → training.wsd ✓
training/pretrain.py → training.numerical_health ✓
training/pretrain.py → training.loss ✓

utils/checkpoint.py → torch, safetensors ✓
utils/checkpoint.py → torch.distributed ✓

models/moe.py → utils/distributed ✓
models/moe.py → ops.triton.grouped_gemm ✓
```

### Issues Identified

1. **No circular imports detected** ✓
2. **Cross-module coupling:** `training/pretrain.py` imports from 9 different modules
3. **Hidden dependencies:** Lazy imports inside methods (e.g., `eval.eval_core`)

### Proposed Fixes

1. **Create dependency graph** in documentation
2. **Enforce import order** via `__init__.py` re-exports
3. **Move lazy imports to module level** where possible

---

## Priority 6: Public API Cleanup

### Current Public API

```python
# training/__init__.py
from .pretrain import Pretrainer, ConfigBundle, main

# utils/__init__.py
from .checkpoint import CheckpointManager

# models/__init__.py
from .transformer import Transformer
```

### Proposed Public API

```python
# training/__init__.py
from .trainer import Pretrainer
from .configs import ConfigBundle, DataConfig, OptimConfig, ...

# training/configs.py
__all__ = ["ConfigBundle", "DataConfig", "OptimConfig", ...]

# utils/checkpoint/__init__.py
from .manager import CheckpointManager
__all__ = ["CheckpointManager"]

# models/moe/__init__.py
from .moe import DeepSeekMoE
__all__ = ["DeepSeekMoE"]
```

---

## Priority 7: Dead Code Audit

### Identified Dead Code

| File | Item | Status |
|------|------|--------|
| `training/pretrain.py` | `Muon` class | Used only if `optimizer == "muon_adamw"` |
| `training/pretrain.py` | `WarmupCosineDecayScheduler` | Used only if `scheduler != "wsd"` |
| `utils/checkpoint.py` | `save_state_dict()` | Used by FSDP path |
| `models/moe.py` | `compute_routing_segments()` | Used by `FusedMoEScatterGather` (not in codebase) |

### Recommended Actions

1. **Keep `Muon`** — valid optimizer choice, not dead
2. **Keep `WarmupCosineDecayScheduler`** — valid scheduler choice, not dead
3. **Keep `save_state_dict()`** — used by FSDP path
4. **Consider removing `compute_routing_segments()`** if no callers exist

---

## Deliverables

### Documentation Files to Create

1. **`docs/MODULARIZATION_PLAN.md`** — This file
2. **`docs/MODULARIZATION_REPORT.md`** — Implementation report with metrics
3. **`docs/DEPENDENCY_GRAPH.md`** — Visual dependency graph
4. **`docs/PUBLIC_API.md`** — Public API reference

### Files to Create

| Priority | File | Lines (est.) |
|----------|------|--------------|
| 1 | `training/configs.py` | 120 |
| 1 | `training/optimization.py` | 200 |
| 1 | `training/dataset.py` | 40 |
| 1 | `training/train_step.py` | 150 |
| 1 | `training/validation.py` | 150 |
| 1 | `training/checkpointing.py` | 100 |
| 1 | `training/curriculum_manager.py` | 80 |
| 1 | `training/monitoring.py` | 100 |
| 1 | `training/trainer.py` | 300 |
| 2 | `utils/checkpoint/atomic.py` | 100 |
| 2 | `utils/checkpoint/metadata.py` | 80 |
| 2 | `utils/checkpoint/retention.py` | 80 |
| 2 | `utils/checkpoint/fsdp.py` | 200 |
| 2 | `utils/checkpoint/dcp.py` | 200 |
| 2 | `utils/checkpoint/recovery.py` | 150 |
| 2 | `utils/checkpoint/async_worker.py` | 100 |
| 2 | `utils/checkpoint/manager.py` | 200 |
| 3 | `models/moe/router.py` | 150 |
| 3 | `models/moe/experts.py` | 80 |
| 3 | `models/moe/balancing.py` | 100 |
| 3 | `models/moe/dispatch.py` | 250 |
| 3 | `models/moe/monitoring.py` | 80 |
| 3 | `models/moe/weight_stacks.py` | 100 |
| 3 | `models/moe/moe.py` | 200 |

**Total new files:** 24
**Total new lines:** ~2,860

### Files to Modify

| File | Change |
|------|--------|
| `training/pretrain.py` | Reduce to ~400 lines |
| `utils/checkpoint.py` | Convert to package |
| `models/moe.py` | Convert to package |
| `training/__init__.py` | Update exports |
| `utils/__init__.py` | Update exports |
| `models/__init__.py` | Update exports |

---

## Required Metrics

### Before Refactoring

| Metric | Value |
|--------|-------|
| Total lines | 7,759 |
| Largest file | 1,195 (pretrain.py) |
| Average file size | 353 lines |
| Files > 500 lines | 3 |
| Function count | ~150 |
| Class count | ~25 |

### After Refactoring (Projected)

| Metric | Value | Change |
|--------|-------|--------|
| Total lines | ~8,500 | +10% (new modules) |
| Largest file | ~400 (trainer.py) | -66% |
| Average file size | ~150 lines | -58% |
| Files > 500 lines | 0 | -100% |
| Function count | ~200 | +33% |
| Class count | ~30 | +20% |

**Note:** Line count increases due to module boundaries, but cognitive complexity decreases dramatically.

---

## Implementation Sequence

### Phase 3.1: Priority 1 — Split pretrain.py (Week 1)

**Day 1-2:**
- Create `training/configs.py`
- Create `training/dataset.py`
- Create `training/optimization.py`

**Day 3-4:**
- Create `training/train_step.py`
- Create `training/validation.py`
- Create `training/checkpointing.py`

**Day 5:**
- Create `training/curriculum_manager.py`
- Create `training/monitoring.py`

**Day 6-7:**
- Create `training/trainer.py`
- Refactor `training/pretrain.py`
- Run all tests

### Phase 3.2: Priority 2 — Split checkpoint.py (Week 2)

**Day 1-2:**
- Create `utils/checkpoint/atomic.py`
- Create `utils/checkpoint/metadata.py`
- Create `utils/checkpoint/retention.py`

**Day 3-4:**
- Create `utils/checkpoint/fsdp.py`
- Create `utils/checkpoint/dcp.py`
- Create `utils/checkpoint/recovery.py`

**Day 5:**
- Create `utils/checkpoint/async_worker.py`
- Create `utils/checkpoint/manager.py`

**Day 6-7:**
- Update `utils/checkpoint/__init__.py`
- Run all tests

### Phase 3.3: Priority 3 — Split moe.py (Week 3)

**Day 1-2:**
- Create `models/moe/router.py`
- Create `models/moe/experts.py`
- Create `models/moe/balancing.py`

**Day 3-4:**
- Create `models/moe/dispatch.py`
- Create `models/moe/monitoring.py`
- Create `models/moe/weight_stacks.py`

**Day 5:**
- Create `models/moe/moe.py`
- Update `models/moe/__init__.py`

**Day 6-7:**
- Run all tests
- Performance regression testing

### Phase 3.4: Priorities 4-7 (Week 4)

**Day 1-2:**
- Consolidate training utilities
- Dependency direction audit

**Day 3-4:**
- Public API cleanup
- Dead code audit

**Day 5-7:**
- Create documentation
- Final metrics collection

---

## Risk Assessment

### High Risk

| Risk | Mitigation |
|------|------------|
| Import cycles | Enforce dependency direction; use lazy imports |
| Test breakage | Run tests after each module extraction |
| Performance regression | Benchmark before/after; profile critical paths |

### Medium Risk

| Risk | Mitigation |
|------|------------|
| API changes | Re-export from `__init__.py` |
| Configuration drift | Keep configs.py as single source |
| Documentation lag | Update docs alongside code |

### Low Risk

| Risk | Mitigation |
|------|------------|
| Merge conflicts | Coordinate with other work streams |
| Code duplication | Extract shared utilities first |

---

## Success Criteria

### Quantitative

- [ ] `training/pretrain.py` < 500 lines
- [ ] `utils/checkpoint.py` split into 6+ modules
- [ ] `models/moe.py` split into 6+ modules
- [ ] All 390+ tests pass
- [ ] No training behavior changes
- [ ] No numerical changes
- [ ] No checkpoint format changes

### Qualitative

- [ ] New engineer can understand codebase in 1 day
- [ ] Clear module ownership
- [ ] Easy to locate functionality
- [ ] Simple to extend
- [ ] Reduced cognitive load

---

## Appendix A: Current File Sizes

| File | Lines |
|------|-------|
| training/pretrain.py | 1,195 |
| utils/checkpoint.py | 1,060 |
| models/moe.py | 785 |
| models/transformer.py | 497 |
| utils/distributed.py | 408 |
| utils/logging.py | 395 |
| models/mla.py | 362 |
| training/numerical_health.py | 356 |
| training/loss.py | 331 |
| models/mtp.py | 288 |
| utils/nccl_profiler.py | 249 |
| utils/compile.py | 240 |
| training/normuon.py | 239 |
| models/gated_deltanet.py | 204 |
| utils/memory_profiler.py | 186 |
| utils/device_setup.py | 170 |
| models/mamba.py | 155 |
| models/rope.py | 154 |
| models/mup.py | 110 |
| training/schedules.py | 104 |
| models/mole.py | 103 |
| training/wsd.py | 82 |
| utils/tensor_checks.py | 64 |

---

## Appendix B: Import Dependencies

```
training/pretrain.py
├── training.configs (new)
├── training.dataset (new)
├── training.optimization (new)
├── training.train_step (new)
├── training.validation (new)
├── training.checkpointing (new)
├── training.curriculum_manager (new)
├── training.monitoring (new)
├── training.normuon
├── training.schedules
├── training.wsd
├── training.numerical_health
├── training.loss
├── utils.checkpoint
├── utils.distributed
├── utils.logging
├── utils.tensor_checks
├── data.async_loader
├── data.curriculum
├── kernels.ce_softcap
├── kernels.linear_relu2
├── models.mtp
└── models.transformer

utils/checkpoint/
├── utils.checkpoint.atomic (new)
├── utils.checkpoint.metadata (new)
├── utils.checkpoint.retention (new)
├── utils.checkpoint.fsdp (new)
├── utils.checkpoint.dcp (new)
├── utils.checkpoint.recovery (new)
└── utils.checkpoint.async_worker (new)

models/moe/
├── models.moe.router (new)
├── models.moe.experts (new)
├── models.moe.balancing (new)
├── models.moe.dispatch (new)
├── models.moe.monitoring (new)
├── models.moe.weight_stacks (new)
└── utils.distributed
```

---

## Appendix C: Test Coverage

### Tests to Verify After Each Phase

| Phase | Test Files |
|-------|------------|
| Priority 1 | `test_smoke.py`, `test_muon.py`, `test_e2e_training.py`, `test_phase2a_reliability.py` |
| Priority 2 | `test_checkpoint.py`, `test_phase2a_reliability.py` |
| Priority 3 | `test_moe.py`, `test_transformer.py` |
| All | Full test suite (`pytest tests/`) |

---

*Plan created: Phase 3 Architectural Modularization*
*Target: Production ML framework maintainability*
