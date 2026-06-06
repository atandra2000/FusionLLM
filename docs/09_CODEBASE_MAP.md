# Codebase Map

## Directory-by-Directory Explanation

### Root Level
- **AGENTS.md**: Opencode agent configuration
- **configs/**: Training configuration files (`pretrain.yaml`, `smoke_pretrain.yaml`)
- **data/**: Data loading and preprocessing utilities
- **docs/**: Documentation (including this project_context directory)
- **eval/**: Evaluation harness integration
- **kernels/**: Custom CUDA kernels for performance optimization
- **models/**: Neural network architectures (Transformer, MLA, MoE, GDN, etc.)
- **ops/**: Low-level operations (primarily Triton kernels)
- **scripts/**: Utility scripts for running experiments
- **tests/**: Unit and integration tests
- **training/**: Training loop and optimizer implementations
- **utils/**: Utility functions (checkpointing, distributed training, logging)
- **requirements.txt**: Python dependencies
- **pyproject.toml**: Project metadata
- **Dockerfile**: Container build instructions
- **Makefile**: Build and test automation
- **prompt.md**: Original task description

### Data Directory (`data/`)
Handles data loading, preprocessing, and curriculum learning:
- **async_loader.py**: Asynchronous sharded data loader for distributed training
- **curriculum.py**: Curriculum learning implementation for switching data mixtures
- **dedup.py**: Deduplication utilities for training data
- **prepare_data.py**: Script to create training data from raw sources (referenced)
- **shard_writer.py**: Utility for creating sharded datasets

### Eval Directory (`eval/`)
Evaluation framework integration:
- **eval_core.py**: Basic perplexity evaluation on token loaders
- **run_lm_eval.py**: Integration with EleutherAI lm-eval-harness for standardized benchmarks

### Kernels Directory (`kernels/`)
Custom CUDA kernels for performance-critical operations:
- **ce_softcap.py**: Fused cross-entropy loss with logit softcap
- **delta_rule.py**: Delta rule kernel (possibly for SSM variants)
- **flash_attn.py**: FlashAttention implementation
- **linear_relu2.py**: Fused linear + squared ReLU operations

### Models Directory (`models/`)
Neural network components:
- **gated_deltanet.py**: Gated Delta Net (GDN) SSM implementation
- **mamba.py**: Mamba-2 SSM implementation (legacy option)
- **mla.py**: Multi-Head Latent Attention (MLA) implementation
- **moe.py**: DeepSeekMoE mixture-of-experts feed-forward network
- **mole.py**: Mixture-of-Experts (MoE) layer implementation
- **mtp.py**: Multi-Token Prediction auxiliary heads
- **mup.py**: μP (μ-transfer) re-initialization utilities
- **rope.py**: Rotary Position Embedding implementation
- **transformer.py**: Main Transformer backbone with hybrid MLA/Mamba schedule

### Ops Directory (`ops/`)
Low-level operations, primarily Triton kernels:
- **triton/grouped_gemm.py**: Grouped GEMM operations for efficient expert computation

### Scripts Directory (`scripts/`)
Utility scripts for execution:
- **bench_mla.py**: Benchmark script for MLA performance
- **run_pretrain_runpod_8xa100.sh**: Script for full pretraining on 8xA100 SXM 80GB
- **run_smoke_8xa100.sh**: Smoke test script for 8xA100 SXM 80GB
- **run_smoke.sh**: Smoke test script for single GPU

### Tests Directory (`tests/`)
Unit and integration tests:
- Tests for all major components: data loading, kernels, models, training utilities
- Includes conftest.py for pytest configuration
- Specific tests for: async_loader, curriculum, checkpoint, distributed, eval, kernels, models, optimizers, schedulers

### Training Directory (`training/`)
Training loop and optimizer implementations:
- **normuon.py**: NorMuon optimizer (Newton-Schulz + Adam moments)
- **pretrain.py**: Main FSDP2 training loop and trainer class
- **schedules.py**: Batch size and sequence length scheduling utilities
- **wsd.py**: Warmup-Stable-Decay (WSD) learning rate scheduler

### Utils Directory (`utils/`)
Shared utility functions:
- **checkpoint.py**: Model checkpointing and resuming logic
- **device_setup.py**: Device initialization utilities
- **distributed.py**: Distributed training setup and FSDP2 wrapping
- **logging.py**: Integration with W&B, MLflow, and CSV logging

## Major Files and Their Responsibilities

### Core Architecture
- **models/transformer.py** (230 lines): Main model backbone that coordinates MLA and GDN layers
- **models/mla.py**: Multi-Head Latent Attention with low-rank KV compression
- **models/moe.py**: DeepSeekMoE with group-limited routing and bias updates
- **models/gated_deltanet.py**: Gated Delta Net SSM implementation
- **models/mamba.py**: Mamba-2 SSM (legacy alternative)
- **models/mtp.py**: Multi-Token Prediction auxiliary heads
- **models/mup.py**: μP re-initialization for stable scaling

### Training Infrastructure
- **training/pretrain.py** (1198 lines): Complete FSDP2 training loop with:
  - Distributed setup and cleanup
  - Model initialization with optional MTP wrapper
  - FSDP2 wrapping via utils/distributed.py
  - Dual optimizer construction (Muon/NorMuon + CautiousAdamW)
  - Learning rate scheduling (WSD or cosine)
  - Gradient accumulation and optimizer stepping
  - Evaluation integration
  - Checkpointing
  - Curriculum learning support
  - Comprehensive logging (W&B, MLflow, CSV, console)

### Distributed Training
- **utils/distributed.py** (294 lines): FSDP2-focused distributed utilities:
  - Process group setup/teardown
  - Collective operations (all-reduce, all-gather, all-to-all)
  - FSDP2 wrapping with per-TransformerBlock granularity
  - Mixed precision configuration (BF16 params, FP32 reduce)
  - Prefetching and resharding tuning options
  - Expert all-to-all operations for MoE routing

### Data Pipeline
- **data/async_loader.py**: Asynchronous sharded data loader supporting:
  - Curriculum learning
  - Dynamic shard selection
  - Micro-prefetching
  - Async mode for overlapping I/O and compute
- **data/curriculum.py**: Two-stage curriculum learning with:
  - Configurable switch step
  - Stage-specific data mixture weights
  - Deterministic shuffling (seed=0)
  - Active shard management
- **data/prepare_data.py**: Data preparation pipeline (referenced in comments)

### Kernels and Optimization
- **kernels/ce_softcap.py**: Fused cross-entropy + logit softcap kernel
- **kernels/linear_relu2.py**: Fused linear + ReLU² kernel
- **kernels/fp8_mla.py**: FP8 matmul kernels for MLA path (Blackwell sm_120)
- **ops/triton/grouped_gemm.py**: Triton kernel for grouped GEMM (efficient expert computation)

### Configuration System
- **configs/pretrain.yaml**: Full training configuration for 8xA100 SXM 80GB runs
- **configs/smoke_pretrain.yaml**: Minimal configuration for debugging and testing
- **training/pretrain.py:ConfigBundle**: Typed configuration structure with nested dataclasses
- **training/pretrain.py:build_config_from_yaml()**: YAML to ConfigBundle converter

### Testing Infrastructure
- **tests/**: Comprehensive unit test suite covering:
  - Data loading (test_async_loader.py, test_loader.py)
  - Kernels (test_ce_softcap.py, test_fp8_mla.py, test_linear_relu2.py)
  - Models (test_mla.py, test_moe.py, test_gdn.py, test_transformer.py)
  - Optimizers (test_muon.py, test_normuon.py)
  - Schedulers (test_schedules.py, test_wsd.py)
  - Distributed utilities (test_distributed.py)
  - Checkpointing (test_checkpoint.py)
  - Curriculum learning (test_curriculum.py, test_curriculum_integration.py)
  - Evaluation (test_eval.py)
  - Logging (test_logging.py)
  - Integration tests (test_pipeline_smoke.py)

## Dependencies Between Files

### Model Dependencies
```
models/transformer.py
├── models/mla.py (Multi-Head Latent Attention)
├── models/moe.py (DeepSeekMoE FFN for MLA layers)
├── models/gated_deltanet.py (GDN SSM for SSM layers)
├── models/mamba.py (Mamba-2 SSM alternative)
├── models/rope.py (Rotary Position Embedding)
└── models/mtp.py (Multi-Token Prediction, optional)
```

### Training Dependencies
```
training/pretrain.py
├── models/transformer.py (model creation)
├── utils/distributed.py (FSDP2 wrapping, collectives)
├── utils/checkpoint.py (model saving/loading)
├── utils/logging.py (W&B, MLflow, CSV logging)
├── training/normuon.py (NorMuon optimizer)
├── training/schedules.py (batch/seq length scheduling)
├── training/wsd.py (WSD scheduler)
├── data/async_loader.py (primary data loading)
├── data/curriculum.py (curriculum learning)
├── eval/eval_core.py (perplexity evaluation)
├── eval/run_lm_eval.py (lm-eval-harness integration)
└── kernels/ (fused kernels when enabled)
```

### Data Pipeline Dependencies
```
data/async_loader.py
├── torch.utils.data.Dataset (base class)
├── data/prepare_data.py (data generation, referenced)
└── data/curriculum.py (curriculum integration)

data/curriculum.py
├── pathlib (manifest handling)
└── torch.utils.data (sampler concepts)
```

### Kernel Dependencies
```
kernels/ce_softcap.py
├── torch.autograd.Function (for custom backward)
└── training/pretrain.py (used when fuse_ce_softcap: true)

kernels/linear_relu2.py
├── torch.autograd.Function (for custom backward)
└── training/pretrain.py (used when fuse_linear_relu2: true)

ops/triton/grouped_gemm.py
├── triton.language and triton.compiler
└── models/moe.py (for expert computation)
```

### Utility Dependencies
```
utils/distributed.py
├── torch.distributed (FSDP2, collectives)
├── models/transformer.py (to find TransformerBlock for wrapping)
└── training/pretrain.py (calls setup, wrap, configure_reshard)

utils/checkpoint.py
├── safetensors (default backend)
├── torch.distributed.checkpoint (dcp backend)
└── training/pretrain.py (calls save/load)

utils/logging.py
├── wandb (optional, when enabled)
├── mlflow (optional, when enabled)
└── training/pretrain.py (calls logging functions)
```

## Call Graph Overview

### Initialization Sequence
1. `training/pretrain.py:main()`
   - Parses arguments
   - Loads YAML config
   - Builds ConfigBundle via `build_config_from_yaml()`
   - Creates `Pretrainer` instance
   - Optionally loads checkpoint
   - Calls `trainer.train()`

2. `training/pretrain.py:Pretrainer.__init__()`
   - Calls `setup_distributed()` (utils/distributed.py)
   - Initializes model: `Transformer()` (models/transformer.py)
   - Optionally wraps with MTP: `MultiTokenPrediction()` (models/mtp.py)
   - Applies FSDP2 wrapping via `wrap_fsdp2()` (utils/distributed.py)
   - Builds optimizers via `_build_optimizers()`
   - Creates learning rate scheduler
   - Sets up batch/sequence length schedulers if enabled
   - Initializes gradient scaler (for FP8)
   - Configures resharding via `configure_reshard()` (utils/distributed.py)
   - Sets up checkpoint manager
   - Initializes curriculum learning if enabled
   - Sets up logging infrastructure

### Training Step Sequence
1. `training/pretrain.py:train()`
   - Sets up data loader (AsyncShardLoader or DataLoader)
   - Handles checkpoint resumption
   - Main training loop:
     - Handles curriculum learning advances
     - Updates batch/sequence length schedulers
     - Gets next batch: `tokens, targets = next(iterate_loader)`
     - Calls `train_step(tokens, targets, micro_step)`

2. `training/pretrain.py:train_step()`
   - Determines if optimizer step needed (gradient accumulation boundary)
   - Applies AMP context (BF16/FP16)
   - Computes forward pass:
     - If MTP enabled: `main_logits, mtp_pairs, _ = self.mtp(tokens)`
     - Else: `logits = self.model(tokens, start_pos=0, use_cache=False)`
   - Computes loss:
     - Main loss: `F.cross_entropy(main_logits, targets)`
     - MTP loss: `self.mtp.compute_mtp_loss(mtp_pairs)` (if MTP)
     - Balance loss: `self._moe_balance_loss()` (MoE auxiliary loss)
     - Combined: `(ce_loss + balance_loss_alpha * balance_loss) / grad_accum`
   - Scales loss and backward: `self.scaler.scale(loss).backward()`
   - If optimizer step:
     - Unsclaes gradients for clipping
     - Applies gradient clipping
     - Steps optimizers (Muon/NorMuon + CautiousAdamW)
     - Updates scheduler
     - Zeroes optimizer gradients
     - Refreshes MoE weight stacks
   - Returns loss metrics

3. `training/pretrain.py:model.forward()` (Transformer)
   - Embedding lookup: `self.embed(tokens)`
   - Processes each TransformerBlock:
     - Applies block (with optional gradient checkpointing)
   - Applies final norm: `self.norm(x)`
   - Applies LM head: `self.head(x)`
   - Applies logit softcap if enabled
   - Applies asymmetric rescale if enabled
   - Returns logits

4. `models/transformer.py:TransformerBlock.forward()`
   - Pre-norm: `self.norm1(x)`
   - Attention/SSM: `self.attn(...)` (either MLA or GDN)
   - Residual connection: `x = x + attn_output`
   - Pre-norm: `self.norm2(x)`
   - FFN: `self.ffn(...)` (either MoE or dense FFN)
   - Residual connection: `x = x + ffn_output`
   - Returns final output

### Evaluation Sequence
1. `training/pretrain.py:_maybe_eval()`
   - Checks if evaluation enabled and at correct interval
   - Checks if main process (rank 0)
   - If `eval_synthetic: true`:
     - Uses `make_synthetic_loader()` (eval/eval_core.py)
   - Else:
     - Would use real validation data loader
   - Calls `run_perplexity()` (eval/eval_core.py)
   - If not synthetic and lm_eval available:
     - Calls `run_lm_eval()` (eval/run_lm_eval.py)
   - Logs results to W&B, MLflow, CSV, and console

### Checkpointing Sequence
1. `training/pretrain.py:save_checkpoint()`
   - Prepares metadata dict (scheduler state, opt_steps, config, tag)
   - If DCP backend and world_size > 1:
     - Calls `ckpt_manager.save_fsdp2_dcp()` (utils/checkpoint.py)
   - Else:
     - If rank 0:
       - Uses `self.raw_model` (unwrapped model)
       - Calls `ckpt_manager.save()` (utils/checkpoint.py)
   - Logs success

2. `training/pretrain.py:load_checkpoint()`
   - If DCP backend and world_size > 1:
     - Calls `ckpt_manager.load_fsdp2_dcp()` (utils/checkpoint.py)
   - Else:
     - Calls `ckpt_manager.load()` (utils/checkpoint.py)
   - Restores scheduler state if present
   - Restores optimizer step count if present
   - Returns the step to resume from

## Key Interfaces and Contracts

### Model Component Interfaces
- **Attention/SSM Modules** (`models/mla.py`, `models/gated_deltanet.py`, `models/mamba.py`):
  - Constructor: `__init__(config, layer_idx, world_size, rank)`
  - Forward: `forward(x) -> y` where shapes `[batch, seq_len, dim]` are preserved
  
- **FFN Modules** (`models/moe.py` for MoE, `training/pretrain.py:DenseFFN` for dense):
  - Forward: `forward(x) -> y` where shapes `[batch, seq_len, dim]` are preserved
  
- **TransformerBlock** (`models/transformer.py:TransformerBlock`):
  - Combines attention/SSM and FFN with pre-norm residuals
  - Forward: `forward(x) -> x` preserving input shape
  
- **Transformer** (`models/transformer.py:Transformer`):
  - Constructor: `__init__(config, world_size=1, rank=0, use_checkpoint=False)`
  - Forward: `forward(tokens, start_pos=0, use_cache=False) -> logits`
    - Input: `[batch, seq_len]` token IDs
    - Output: `[batch, seq_len, vocab_size]` logits

### Distributed Training Contracts
- **FSDP2 Wrapping** (`utils/distributed.py:wrap_fsdp2()`):
  - Expects model with `TransformerBlock` submodules for per-block wrapping
  - Handles mixed precision via `param_dtype` and `reduce_dtype`
  - Applies backward prefetch and gather limiting as configured
  
- **Collective Operations** (`utils/distributed.py`):
  - `all_reduce_mean(tensor)`: Returns mean across ranks
  - `all_gather(tensor)`: Returns list of tensors from all ranks
  - `is_main_process()`: True only for rank 0
  - Barrier and all-to-all operations for expert parallelism

### Optimizer Interface
- **Dual Optimizer System** (`training/pretrain.py:_build_optimizers()`):
  - Separates parameters into:
    - Matrix parameters (ndim ≥ 2, not embed/head): Muon/NorMuon
    - Remaining parameters: CautiousAdamW
  - Returns tuple: `(primary_optimizer, adamw_optimizer)` where primary may be None
  - Both optimizers support standard PyTorch optimizer interface:
    - `step()`, `zero_grad()`, `param_groups`, `state_dict()`, `load_state_dict()`

### Data Loader Contracts
- **AsyncShardLoader** (`data/async_loader.py`):
  - Constructor: `AsyncShardLoader(manifest_path, batch_size, grad_accum, seqlen, rank, world_size, ...)`
  - Methods: 
    - `start()`: Begin asynchronous loading
    - `set_batch_size(new_bs)`: Update batch size
    - `set_seq_len(new_seqlen)`: Update sequence length
    - `set_shards(active_shards)`: Update active shards for curriculum
    - Implements iterator protocol: `__iter__()`, `__next__()`
    
- **Standard DataLoader** (`training/pretrain.py`):
  - Uses `PretrainDataset` with optional `DistributedSampler`
  - Yields `(tokens, targets)` tensors
  - `targets = torch.roll(tokens, shifts=-1, dims=1)` (next-token prediction)

## Inheritance and Composition Patterns

### Composition Over Inheritance
The codebase favors composition:
- **Transformer** composes `ParallelEmbedding`, `TransformerBlock` list, `RMSNorm`, `Linear` head
- **TransformerBlock** composes attention/SSM module and FFN module
- **Attention/SSM modules** compose linear projections and operations
- **FFN modules** compose linear projections and activation functions
- **Pretrainer** composes model, optimizers, schedulers, data loader, checkpoint manager, logger

### Minimal Inheritance
Limited use of inheritance, primarily for:
- **Optimizer classes**: `Muon`, `NorMuon` inherit from `torch.optim.Optimizer`
- **Scheduler classes**: `WarmupCosineDecayScheduler`, `WSD scheduler` inherit from `_LRScheduler`
- **Module classes**: All `nn.Module` subclasses inherit from `torch.nn.Module`
- **Custom autograd functions**: Kernel implementations inherit from `torch.autograd.Function`

### Configuration Composition
- **ConfigBundle** uses composition of dataclasses:
  - `DataConfig`, `OptimConfig`, `ScheduleConfig`, `EvalConfig`, `CheckpointConfig`, `LoggingConfig`
  - Each handles a specific concern with clear boundaries
  - Easy to extend with new configuration sections

## Data Flow Summary

1. **Configuration**: YAML files → `ConfigBundle` → controls all behavior
2. **Data**: Token files → `AsyncShardLoader`/`DataLoader` → `(tokens, targets)` batches
3. **Model Embedding**: `ParallelEmbedding` → lookups → `[batch, seq_len, dim]` embeddings
4. **Transformer Blocks**: Sequence of blocks applying:
   - Pre-norm → Attention/SSM → Add & Norm → FFN → Add
5. **LM Head**: Final norm → Linear projection → `[batch, seq_len, vocab_size]` logits
6. **Loss Computation**: Logits + targets → cross-entropy → scalar loss
7. **Backward Pass**: Gradients flow back through same path in reverse
8. **Optimizer Step**: Parameter updates based on gradients
9. **Checkpointing**: Periodic saving of model, optimizer, scheduler states
10. **Evaluation**: Periodic inference on validation data to compute perplexity and task scores
11. **Logging**: Metrics sent to W&B, MLflow, CSV, and console (rank 0 only)

## Hot Paths (Frequently Executed Code)
1. **Inner Training Loop** (`training/pretrain.py:train_step()`):
   - Embedding lookup
   - Transformer block processing (30 iterations)
   - Loss computation
   - Backward pass
   - Gradient clipping and optimizer stepping (every grad_accum steps)

2. **Forward Pass Through TransformerBlock** (`models/transformer.py:TransformerBlock._forward()`):
   - Two RMSNorm operations
   - Attention/SSM computation (MLA or GDN)
   - FFN computation (MoE or dense)
   - Two residual additions

3. **MLA Computation** (`models/mla.py:MultiHeadLatentAttention.forward()`):
   - Low-rank QKV projections
   - RoPE application
   - Attention score computation
   - Mixed precision GEMM operations
   - Output projection

4. **MoE Computation** (`models/moe.py:DeepSeekMoE.forward()`):
   - Gate score computation
   - Expert routing and capacity enforcement
   - ExpertFFN computation (sparse activation)
   - Result combination

5. **Kernel Operations** (when enabled):
    - Fused CE + softcap (`kernels/ce_softcap.py`)
    - Fused linear + ReLU² (`kernels/linear_relu2.py`)
    - Grouped GEMM for experts (`ops/triton/grouped_gemm.py`)

## Cold Paths (Infrequently Executed Code)
1. **Initialization**:
   - Model construction and parameter initialization
   - FSDP2 wrapping and configuration
   - Optimizer and scheduler setup
   - Data loader initialization
   - Logging infrastructure setup

2. **Evaluation**:
   - Perplexity computation on validation data
   - LM-eval-harness task execution
   - Metric logging to external services

3. **Checkpointing**:
   - Model and optimizer state serialization
   - File I/O for checkpoint storage
   - Metadata preparation and storage

4. **Curriculum Learning**:
   - Stage switching checks (every step)
   - Shard set updates (only at switch points)
   - Data loader reconfiguration (infrequent)

5. **Error Handling**:
   - Exception paths in logging, checkpointing, data loading
   - Numerical error detection (NaN/Inf checks)
   - Distributed training error handling

This codebase map provides a comprehensive overview of the repository structure, dependencies, and execution patterns to help future agents navigate and modify the code effectively.