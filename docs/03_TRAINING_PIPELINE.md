# Training Pipeline

## End-to-End Training Flow
The training pipeline follows this sequence:
1. **Configuration Loading**: Loads YAML config (`configs/pretrain.yaml`) and command-line arguments
2. **Distributed Setup**: Initializes PyTorch distributed training via `setup_distributed()`
3. **Model Initialization**: Creates Transformer model with optional MTP wrapper
4. **FSDP2 Wrapping**: Wraps model with `torch.distributed.fsdp.fully_shard` for ZeRO-3 style sharding
5. **Optimizer Setup**: Creates Muon/NorMuon for matrix parameters + CautiousAdamW for others
6. **Learning Rate Scheduling**: Uses WSD (Warmup-Stable-Decay) or Cosine scheduler
7. **Data Loading**: Uses either AsyncShardLoader (with manifest) or standard DataLoader
8. **Training Loop**: Iterates through steps, accumulating gradients, performing optimizer steps
9. **Evaluation**: Runs perplexity and LM-eval-harness tasks at intervals
10. **Checkpointing**: Saves model state periodically
11. **Curriculum Learning**: Optionally switches data mixtures mid-training
12. **Logging**: Tracks metrics to W&B, MLflow, and CSV files

## Dataset Preparation
### Synthetic Data Fallback
- If `data/pretrain_data.bin` missing, generates random token data (1M tokens)
- Persists generated data for future runs
- Used for testing and development

### Real Data Pipeline
- Primary path: `data/pretrain_data.bin` (packed token tensor)
- Validation path: `data/validation_data.bin`
- Data mixing configuration in `configs/pretrain.yaml:data.data_mix`:
  - fineweb_edu: 0.60
  - finemath: 0.15
  - stack_edu: 0.15
  - cosmopedia: 0.05
  - openr1_math: 0.05

### Data Loading Mechanisms
1. **AsyncShardLoader** (`data/async_loader.py`):
   - Used when `shard_manifest_path` is provided
   - Supports curriculum learning and dynamic shard selection
   - Features: micro-prefetching, async mode, rank-based sharding

2. **Standard DataLoader** (`training/pretrain.py:train()`):
   - Uses `PretrainDataset` class
   - Implements packed sequence dataset: each sample contains consecutive tokens
   - Supports distributed sampling via `DistributedSampler`

### Dataset Class Details (`data/async_loader.py` and `training/pretrain.py`)
- `PretrainDataset`: Creates overlapping windows of `max_seq_len + 1` tokens
- Input: tokens[i:i+L], Target: tokens[i+1:i+L+1] (next-token prediction)
- Handles sharding across DP ranks when world_size > 1

## Curriculum Learning
Implemented in `data/curriculum.py`:
- **Two-stage switching**: Controlled by `curriculum_switch_step`
- **Stage 1** (default): Web-heavy mixture (fineweb_edu: 0.70, stack_edu: 0.15, etc.)
- **Stage 2** (after switch): Code/math-heavy (fineweb_edu: 0.30, stack_edu: 0.25, openr1_math: 0.25)
- **Weights Configuration**: Overridable via `curriculum_stage1_weights` and `curriculum_stage2_weights`
- **Mechanism**: 
  - Curriculum object tracks current stage
  - `advance(step)` returns True when switch occurs
  - Active shards communicated to loader via `set_shards()`
  - Seed-based reproducibility (seed=0)

## FSDP Strategy
### Implementation (`utils/distributed.py:wrap_fsdp2`)
- Uses `torch.distributed.fsdp.fully_shard` (FSDP2)
- **Parameter Groups**: Automatic per-layer wrapping
- **Sharding Strategies**:
  - `FULL_SHARD`: Shards parameters, gradients, optimizer states (default)
  - `SHARD_GRAD_OP`: Shards only during gradient operations
  - `NO_SHARD`: No sharding (debugging)
- **Settings from config**:
  - `fsdp_param_dtype: bf16` (parameter storage dtype)
  - `fsdp_reduce_dtype: fp32` (reduction dtype for numerical stability)
  - `fsdp_forward_prefetch: false` (saves H2D bandwidth)
  - `fsdp_backward_prefetch: true` (standard for performance)
  - `fsdp_limit_all_gathers: true` (prevents NCCL queue buildup)
  - `shard_keep_last: 1` (keeps last N FSDP units resident to reduce resharding)

### Memory Characteristics
- **Static State**: ~4.5 GB/GPU on 8×A100 80GB SXM (as noted in config comments)
- **Activation Memory**: Managed via gradient checkpointing
- **KV Cache**: Dynamically allocated during forward pass

## Checkpointing
### Implementation (`utils/checkpoint.py`)
- **Backends**:
  - `safetensors`: Default, CPU-safe format
  - `dcp`: PyTorch Distributed Checkpoint (for FSDP2 with world_size > 1)
- **Frequency**: Controlled by `checkpoint.save_every` (default: 1000 steps)
- **Metadata Saved**:
  - Scheduler state
  - Optimizer step count (`_opt_steps`)
  - Training configuration
  - Step number and optional tag
- **Loading**: Restores model, optimizer, scheduler states
- **Async Option**: Controlled by `hardware.async_checkpointing` (default: true)

## Logging
### Systems Used
1. **Weights & Biases** (`wandb_enabled: true`):
   - Project: `fusionllm-pretrain`
   - Logs: loss, learning rates, MoE routing stats, system metrics
   - Tags: Empty by default

2. **MLflow** (`mlflow_enabled: true`):
   - Tracking URI: `file:./mlruns` (local file store)
   - Experiment: `fusionllm-pretrain`
   - Tags: Empty dict by default

3. **Console Logging**:
   - Rank 0 only prints to avoid duplication
   - Regular interval: `log_interval: 50` steps
   - Includes: loss, learning rates, Muon LR (if applicable)

4. **Runs CSV Logger** (`utils/logging.py:RunsCsvLogger`):
   - Persists metrics to CSV for post-processing
   - Logs: step, loss, perplexity, eval task scores

### MoE-Specific Logging
- Routing statistics logged every 200 steps when rank 0
- Includes: expert counts, load balancing metrics, routing entropy

## Optimizers
### Dual-Optimizer Strategy
1. **Matrix Parameters** (weights with ndim ≥ 2, excluding embed/head):
   - **Normuon**: Default (`optimizer: normuon_adamw`)
     - Combines Newton-Schulz orthogonalization with Adam-style moment tracking
     - Uses `betas` from adamw_betas: [0.9, 0.95]
     - Weight decay applied directly
   - **Muon** (alternative): Pure Newton-Schulz with momentum
     - Momentum: `muon_momentum: 0.95`
     - Learning rate: `muon_lr: 0.02`

2. **Non-Matrix Parameters** (embeddings, norms, biases, LM head):
   - **CautiousAdamW**:
     - Base LR: `lr: 3e-4`
     - Betas: `adamw_betas: [0.9, 0.95]`
     - Weight decay: `weight_decay: 0.1`
     - **Cautious masking**: Only applies WD where grad*weight > 0 (prevents divergence)
     - Epsilon: Default AdamW epsilon (1e-8)

### Optimizer Initialization (`training/pretrain.py:_build_optimizers`)
- Separates parameters by: `is_matrix and not is_embed_or_head`
- Logs parameter counts for each optimizer on rank 0
- Uses fused CUDA kernels when available

## Schedulers
### Learning Rate Scheduling
1. **WSD (Warmup-Stable-Decay)** - Default (`scheduler: wsd`):
   - Warmup: `wsd_warmup_frac: 0.01` (500 steps of 50k)
   - Stable: `wsd_stable_frac: 0.84` (42k steps at max LR)
   - Decay: `wsd_decay: linear` (remaining 7.5k steps to min LR)
   - Min LR Ratio: `min_lr_ratio: 0.1` (LR decays to 10% of max)

2. **Warmup-Cosine-Decay** (alternative):
   - Warmup: `warmup_steps: 500` (linear increase)
   - Cosine decay: to `min_lr_ratio * max_lr`
   - Flat minimum after decay completion

### Batch Size and Sequence Length Scheduling
- **Batch Size Schedule** (`training/schedules.py:BatchSizeSchedule`):
  - Enabled by `batch_size_schedule_enabled: false`
  - Linear ramp from `initial_batch_size: 2` to `final_batch_size: 8` over `batch_size_schedule_steps: 5000`
  
- **Sequence Length Schedule** (`training/schedules.py:SeqLenSchedule`):
  - Enabled by `seq_len_schedule_enabled: false`
  - Linear ramp from `initial_seq_len: 2048` to `final_seq_len: 8192` over `seq_len_schedule_steps: 5000`

## Evaluation
### Configuration (`configs/pretrain.yaml:training`)
- `eval_enabled: false` (disabled by default for faster iteration)
- `eval_interval: 1000` steps
- `eval_max_batches: 8` batches per evaluation
- `eval_synthetic: true` (uses synthetic data loader)
- **Tasks** (when `eval_synthetic: false`):
  - hellaswag (commonsense reasoning)
  - arc_challenge (science reasoning)
  - piqa (physical commonsense)
  - winogrande (coreference resolution)
  - boolq (question answering)

### Implementation
1. **Synthetic Evaluation** (`eval/eval_core.py`):
   - Creates random token batches for perplexity measurement
   - Measures: loss and perplexity on next-token prediction

2. **LM-Eval Harness** (`eval/run_lm_eval.py`):
   - Integrates with EleutherAI lm-eval-harness
   - Runs specified tasks with 50-example limit
   - Requires `lm_eval` package and CUDA

### Logging
- Validation metrics logged to W&B, MLflow, and CSV
- Includes: perplexity, loss, and individual task scores

## Performance Characteristics
### Estimated Throughput
- From config comments: ≈4.0M tokens/sec on 8×A100 SXM 80GB
- 150B tokens ≈ 10.4 hours wall-clock for full training
- Per-step tokens: 1,048,576 ≈ 1M tokens/opt-step

### Memory Efficiency Techniques
- **Gradient Checkpointing**: `use_checkpoint: true` (activations recomputed in backward)
- **BF16 Precision**: `dtype: bf16` for reduced memory bandwidth
- **FSDP Sharding**: ~4.5 GB/GPU static state on 8×A100 80GB SXM
- **LoRA Compression in MLA**: Reduces KV cache size significantly
- **MoE Sparsity**: Only ~15.6% of FFN parameters active per token

## Key Files
- Main training loop: `training/pretrain.py`
- Configuration builder: `training/pretrain.py:build_config_from_yaml()`
- Dataset loading: `data/async_loader.py`, `training/pretrain.py:PretrainDataset`
- Optimizers: `training/normuon.py`, `training/pretrain.py:_build_optimizers()`
- Schedulers: `training/schedules.py`, `training/wsd.py`
- Distributed setup: `utils/distributed.py`
- Checkpointing: `utils/checkpoint.py`
- Logging: `utils/logging.py`
- Evaluation: `eval/eval_core.py`, `eval/run_lm_eval.py`