# Dependency Graph — FusionLLM Codebase

## Overview

This document visualizes the dependency relationships between modules in the FusionLLM codebase. The goal is to enforce clean dependency flow and prevent circular imports.

## Preferred Dependency Direction

```
configs (dataclasses)
    ↓
models (neural network modules)
    ↓
training (training logic)
    ↓
entrypoints (CLI, scripts)
```

## Current Dependency Graph

### Level 0: No Dependencies (Foundation)

```
utils/tensor_checks.py
utils/device_setup.py
training/schedules.py
training/wsd.py
```

### Level 1: Depends on Level 0

```
utils/distributed.py → utils/device_setup.py
utils/logging.py → (stdlib only)
utils/compile.py → (torch only)
utils/memory_profiler.py → (torch only)
utils/nccl_profiler.py → (torch, utils/distributed)
training/normuon.py → (torch only)
training/loss.py → (torch only)
```

### Level 2: Depends on Level 0-1

```
utils/checkpoint.py → utils/distributed.py
                     → utils/logging.py

training/numerical_health.py → utils/tensor_checks.py

models/rope.py → (torch only)
models/mup.py → (torch only)
models/mamba.py → (torch only)
models/gated_deltanet.py → (torch only)
models/mla.py → utils/distributed.py
              → models/rope.py
models/moe.py → utils/distributed.py
              → ops/triton/grouped_gemm.py (optional)
```

### Level 3: Depends on Level 0-2

```
models/transformer.py → models/mla.py
                      → models/moe.py
                      → models/mamba.py
                      → models/gated_deltanet.py

models/mtp.py → models/transformer.py
```

### Level 4: Depends on Level 0-3

```
training/pretrain.py → training/normuon.py
                     → training/schedules.py
                     → training/wsd.py
                     → training/numerical_health.py
                     → training/loss.py
                     → utils/checkpoint.py
                     → utils/distributed.py
                     → utils/logging.py
                     → utils/tensor_checks.py
                     → models/mtp.py
                     → models/transformer.py
                     → data/async_loader.py
                     → data/curriculum.py
```

### Level 5: Entrypoints

```
scripts/*.py → training/pretrain.py
tests/*.py → (various modules)
```

## Visual Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRYPOINTS                              │
│  scripts/pretrain.py, scripts/eval.py, tests/*.py              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ training/pretrain.py (1,195 lines)                       │  │
│  │   ├── ConfigBundle, DataConfig, OptimConfig, ...         │  │
│  │   ├── Pretrainer class                                   │  │
│  │   ├── WarmupCosineDecayScheduler                         │  │
│  │   ├── Muon, CautiousAdamW                                │  │
│  │   └── build_config_from_yaml, main                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │ numerical_health │ │     loss.py     │ │   normuon.py    │  │
│  │    (356 lines)   │ │   (331 lines)   │ │  (239 lines)    │  │
│  └────────┬────────┘ └─────────────────┘ └─────────────────┘  │
│           │                                                      │
│  ┌────────▼────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │  tensor_checks   │ │  schedules.py   │ │     wsd.py      │  │
│  │   (64 lines)     │ │  (104 lines)    │ │  (82 lines)     │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODELS LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                transformer.py (497 lines)                │  │
│  │   Transformer, TransformerBlock, DenseFFN, ...           │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────┼─────────────────────────────────┐  │
│  │                        │                                  │  │
│  ▼                        ▼                                  │  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │    mla.py       │ │    moe.py       │ │   mamba.py      │  │
│  │  (362 lines)    │ │  (785 lines)    │ │  (155 lines)    │  │
│  └────────┬────────┘ └────────┬────────┘ └─────────────────┘  │
│           │                    │                                │
│  ┌────────▼────────┐ ┌────────▼────────┐                      │
│  │    rope.py      │ │   distributed   │                      │
│  │  (154 lines)    │ │  (408 lines)    │                      │
│  └─────────────────┘ └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        UTILS LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ checkpoint.py (1,060 lines)  │  distributed.py (408)     │  │
│  │ logging.py (395 lines)       │  compile.py (240)         │  │
│  │ device_setup.py (170)        │  memory_profiler.py (186) │  │
│  │ nccl_profiler.py (249)       │  tensor_checks.py (64)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FOUNDATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PyTorch, CUDA, NCCL, safetensors, triton (optional)      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Problematic Dependencies (None Found)

### Circular Imports
**Status:** ✅ No circular imports detected

### Cross-Module Coupling
**Status:** ⚠️ `training/pretrain.py` depends on 9+ modules

**Recommendation:** Split into focused modules (Priority 1)

### Hidden Dependencies
**Status:** ⚠️ Lazy imports inside methods

Examples:
- `training/pretrain.py:659` → `from eval.eval_core import ...`
- `training/pretrain.py:681` → `from eval.run_lm_eval import ...`
- `training/pretrain.py:610` → `from utils.logging import RunsCsvLogger`

**Recommendation:** Move to module level or document rationale

## Post-Refactoring Dependency Graph

### Level 0: No Dependencies

```
utils/tensor_checks.py
utils/device_setup.py
training/schedules.py
training/wsd.py
```

### Level 1: Depends on Level 0

```
utils/distributed.py → utils/device_setup.py
utils/logging.py → (stdlib only)
utils/compile.py → (torch only)
utils/memory_profiler.py → (torch only)
utils/nccl_profiler.py → (torch, utils/distributed)
training/normuon.py → (torch only)
training/loss.py → (torch only)
```

### Level 2: Depends on Level 0-1

```
utils/checkpoint/atomic.py → (torch, safetensors)
utils/checkpoint/metadata.py → (json, logging)
utils/checkpoint/retention.py → (pathlib, logging)
utils/checkpoint/fsdp.py → (torch.distributed)
utils/checkpoint/dcp.py → (torch.distributed.checkpoint)
utils/checkpoint/recovery.py → (safetensors, torch)
utils/checkpoint/async_worker.py → (threading, queue)
utils/checkpoint/manager.py → utils/checkpoint/*

training/numerical_health.py → utils/tensor_checks.py
training/configs.py → (dataclasses)
training/dataset.py → (torch, os)

models/rope.py → (torch only)
models/mup.py → (torch only)
models/mamba.py → (torch only)
models/gated_deltanet.py → (torch only)
models/moe/router.py → (torch, torch.nn)
models/moe/experts.py → (torch, torch.nn, torch.distributed)
models/moe/balancing.py → (torch only)
models/moe/dispatch.py → (torch, torch.distributed)
models/moe/monitoring.py → (torch only)
models/moe/weight_stacks.py → (torch only)
models/mla.py → utils/distributed.py, models/rope.py
```

### Level 3: Depends on Level 0-2

```
training/optimization.py → training/normuon.py
training/curriculum_manager.py → data/curriculum.py
training/monitoring.py → training/numerical_health.py, utils/logging.py

models/moe/moe.py → models/moe/router.py
                   → models/moe/experts.py
                   → models/moe/balancing.py
                   → models/moe/dispatch.py
                   → models/moe/monitoring.py
                   → models/moe/weight_stacks.py

models/transformer.py → models/mla.py, models/moe/moe.py, ...
models/mtp.py → models/transformer.py
```

### Level 4: Depends on Level 0-3

```
training/train_step.py → training/optimization.py
                        → utils/tensor_checks.py
                        → utils/distributed.py

training/validation.py → eval/eval_core.py, eval/run_lm_eval.py

training/checkpointing.py → utils/checkpoint/manager.py
                           → utils/distributed.py

training/trainer.py → training/train_step.py
                     → training/validation.py
                     → training/checkpointing.py
                     → training/curriculum_manager.py
                     → training/monitoring.py
```

### Level 5: Thin Entrypoint

```
training/pretrain.py → training/trainer.py
                     → training/configs.py
                     → training/dataset.py
```

### Level 6: Scripts

```
scripts/pretrain.py → training/pretrain.py
tests/*.py → (various modules)
```

## Dependency Rules

### 1. No Backward Dependencies
```
Level N modules must NOT import from Level N+1 or higher
```

### 2. Minimal Cross-Level Dependencies
```
Prefer importing from adjacent levels
Avoid jumping 2+ levels
```

### 3. Stable Foundation
```
Level 0-1 modules must NOT change frequently
Level 0-1 modules must NOT depend on training logic
```

### 4. Clear Ownership
```
Each module has a single owner
Changes to shared modules require review
```

## Import Order Convention

```python
# 1. Standard library
import os
import json
from pathlib import Path

# 2. Third-party
import torch
import torch.nn as nn

# 3. Project-level (foundation first)
from utils.distributed import setup_distributed
from utils.logging import get_logger
from utils.checkpoint import CheckpointManager

# 4. Project-level (training)
from training.configs import ConfigBundle
from training.normuon import NorMuon

# 5. Project-level (models)
from models.transformer import Transformer
from models.moe import DeepSeekMoE
```

## Testing Dependencies

### Unit Tests (No training loop)
```
tests/test_moe.py → models/moe.py
tests/test_transformer.py → models/transformer.py
tests/test_loss.py → training/loss.py
tests/test_normuon.py → training/normuon.py
tests/test_numerical_health.py → training/numerical_health.py
tests/test_checkpoint.py → utils/checkpoint.py
```

### Integration Tests (Full training)
```
tests/test_smoke.py → training/pretrain.py
tests/test_e2e_training.py → training/pretrain.py
tests/test_phase2a_reliability.py → training/pretrain.py
```

---

*Dependency graph created: Phase 3 Architectural Modularization*
