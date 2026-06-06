# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the project, organized by section with default values, recommended values, and notes on interactions.

## Configuration Hierarchy
Configuration is loaded from YAML files (default: `configs/pretrain.yaml`) and can be overridden via command-line arguments. The configuration is structured into sections that map to `ConfigBundle` dataclasses in `training/pretrain.py`.

## Training Configuration (`training:`)

### Loop Parameters
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `micro_batch_size` | 2 | 2-8 (per rank) | Batch size per GPU before gradient accumulation |
| `gradient_accumulation_steps` | 16 | 8-32 | Steps to accumulate gradients before optimizer update |
| `total_steps` | 50_000 | 10k-500k | Total training steps |
| `warmup_steps` | 500 | 100-1000 | Linear warmup steps for learning rate |

*Interaction*: Effective batch size = `micro_batch_size × gradient_accumulation_steps × world_size`

### Optimization
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `lr` | 3e-4 | 1e-4 - 5e-4 | AdamW base learning rate (non-matrix params) |
| `muon_lr` | 0.02 | 0.01 - 0.1 | Muon/NorMuon learning rate (matrix params) |
| `muon_momentum` | 0.95 | 0.9 - 0.99 | Momentum for Muon optimizer |
| `adamw_betas` | [0.9, 0.95] | [0.9, 0.95]-[0.95, 0.99] | Betas for AdamW component |
| `min_lr_ratio` | 0.1 | 0.05 - 0.2 | Minimum LR as fraction of max LR |
| `weight_decay` | 0.1 | 0.01 - 0.2 | Weight decay coefficient |
| `cautious_wd` | true | true | Enable cautious weight decay (sign-masked) |
| `grad_clip` | 1.0 | 0.5 - 2.0 | Gradient clipping norm |

*Interaction*: 
- Matrix parameters (weight tensors with ndim≥2, excluding embed/head) use Muon/NorMuon
- All other parameters use CautiousAdamW
- `cautious_wd` only affects the AdamW optimizer

### Precision and Hardware
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `dtype` | bf16 | bf16 | Computation dtype (bf16, fp16) |
| `use_checkpoint` | true | true | Enable gradient checkpointing per block |
| `use_fa3` | false | false (opt-in) | Flash-Attention 3 (BF16 fallback on CPU) |

*Interaction*: 
- Gradient checkpointing trades compute for memory savings (~30-40% activation reduction)

### Logging and Monitoring
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `save_dir` | checkpoints/pretrain | checkpoints/pretrain | Directory for checkpoints |
| `save_interval` | 500 | 100-1000 | Steps between checkpoints |
| `log_interval` | 50 | 10-100 | Steps between console/logging updates |
| `wandb_enabled` | true | true | Enable Weights & Biases logging |
| `wandb_project` | fusionllm-pretrain | - | W&B project name |
| `mlflow_enabled` | true | true | Enable MLflow logging |
| `mlflow_tracking_uri` | file:./mlruns | - | MLflow tracking destination |
| `mlflow_experiment_name` | fusionllm-pretrain | - | MLflow experiment name |

*Interaction*: 
- Logging only occurs on rank 0 to avoid duplication
- W&B and MLflow can be enabled independently
- Console logs always enabled for rank 0

### Evaluation
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `eval_enabled` | false | true (for production) | Enable evaluation during training |
| `eval_interval` | 1000 | 500-2000 | Steps between evaluations |
| `eval_max_batches` | 8 | 4-16 | Max batches per evaluation run |
| `eval_synthetic` | true | false (for real eval) | Use synthetic data (fast) vs real data |
| `eval_tasks` | [hellaswag, arc_challenge, piqa, winogrande, boolq] | - | LM-eval harness tasks |

*Interaction*: 
- When `eval_synthetic: true`, uses deterministic random loader
- When `eval_synthetic: false`, requires `lm_eval` package and real validation data
- Evaluation only runs on rank 0

### Curriculum Learning
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `curriculum_switch_step` | 0 | 10000+ | Step to switch data mixtures (0=disabled) |
| `curriculum_stage1_weights` | None | see below | Stage 1 data mixture weights |
| `curriculum_stage2_weights` | None | see below | Stage 2 data mixture weights |

*Default Stage 1 Weights*:
- fineweb_edu: 0.70
- stack_edu: 0.15
- openr1_math: 0.05
- fineweb2: 0.10

*Default Stage 2 Weights*:
- fineweb_edu: 0.30
- stack_edu: 0.25
- openr1_math: 0.25
- fineweb2: 0.10
- smollm_corpus: 0.10

*Interaction*: 
- Requires `shard_manifest_path` to be set in data config
- Switch occurs exactly at `curriculum_switch_step`
- Uses seed=0 for reproducible shuffling

### Loss Balancing
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `balance_loss_alpha` | 1e-4 | 1e-5 - 1e-3 | MoE load balancing loss coefficient |
| `bias_update_speed` | 1e-3 | 1e-4 - 1e-2 | Expert bias update rate |
| `bias_update_every` | 10 | 5-20 | Steps between bias updates |

*Interaction*: 
- Load balancing loss = `balance_loss_alpha × sum(expert_aux_losses)`
- Bias updates use exponential moving average: `bias += speed × (target_fraction - actual_fraction)`
- Target fraction = 1.0 / n_routed_experts per expert

### FSDP2 Configuration
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `fsdp_shard_strategy` | FULL_SHARD | FULL_SHARD | Sharding strategy (FULL_SHARD\|SHARD_GRAD_OP\|NO_SHARD) |
| `fsdp_forward_prefetch` | false | false | Enable forward parameter prefetch |
| `fsdp_backward_prefetch` | true | true | Enable backward parameter prefetch |
| `fsdp_limit_all_gathers` | true | true | Limit NCCL queue depth for all-gather |
| `fsdp_param_dtype` | bf16 | bf16 | Parameter storage dtype |
| `fsdp_reduce_dtype` | fp32 | fp32 | Gradient reduction dtype |
| `shard_keep_last` | 1 | 1-2 | Keep last N FSDP units resident after forward |

*Interaction*: 
- FULL_SHARD shards parameters, gradients, and optimizer states
- SHARD_GRAD_OP only shards during gradient operations (saves memory but increases compute)
- NO_SHARD disables sharding (single-device behavior)
- `fsdp_reduce_dtype: fp32` provides numerical stability for gradient reductions

### Checkpointing
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `checkpoint_backend` | safetensors | safetensors\|dcp | Checkpoint format (safetensors\|dcp) |
| `save_every` | 1000 | 500-2000 | Steps between checkpoints (alias for save_interval) |

*Interaction*: 
- `safetensors`: CPU-safe, portable format
- `dcp`: PyTorch Distributed Checkpoint (requires world_size>1 for full benefits)
- Asynchronous checkpointing controlled by `hardware.async_checkpointing`

### Optimizer and Scheduler Selection
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `optimizer` | normuon_adamw | normuon_adamw\|muon_adamw | Optimizer pair type |
| `scheduler` | wsd | wsd\|cosine | Learning rate scheduler |
| `wsd_warmup_frac` | 0.01 | 0.005-0.02 | Warmup fraction (of total_steps) |
| `wsd_stable_frac` | 0.84 | 0.7-0.9 | Stable fraction (of total_steps) |
| `wsd_decay` | linear | linear\|cosine | Decay schedule type |

*Interaction*: 
- `normuon_adamw`: NorMuon (Newton-Schulz + Adam moments) + CautiousAdamW
- `muon_adamw`: Pure Muon (Newton-Schulz + momentum) + CautiousAdamW
- WSD: Linear warmup → stable LR → linear/cosine decay to min_lr_ratio
- Cosine: Linear warmup → cosine decay → flat min_lr_ratio

### Batch and Sequence Length Scheduling
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `batch_size_schedule_enabled` | false | false\|true | Enable batch size scheduling |
| `initial_batch_size` | 2 | 1-4 | Starting batch size |
| `final_batch_size` | 8 | 4-16 | Ending batch size |
| `batch_size_schedule_steps` | 5000 | 1000-10000 | Steps for batch size transition |
| `seq_len_schedule_enabled` | false | false\|true | Enable sequence length scheduling |
| `initial_seq_len` | 2048 | 1024-4096 | Starting sequence length |
| `final_seq_len` | 8192 | 4096-16384 | Ending sequence length |
| `seq_len_schedule_steps` | 5000 | 1000-10000 | Steps for seq length transition |

*Interaction*: 
- Linear scheduling between initial and final values
- When enabled, overrides static `batch_size` and `max_seq_len` in data config
- Requires loader to support dynamic resizing (AsyncShardLoader does, standard DataLoader does not)

### Kernel Fusion
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `fuse_ce_softcap` | true | true | Fuse cross-entropy loss with logit softcap |
| `fuse_linear_relu2` | true | true | Fuse linear + squared ReLU operations |

*Interaction*: 
- Reduces kernel launches and memory bandwidth
- `fuse_ce_softcap`: Combines `F.cross_entropy` and `logits = cap * tanh(logits / cap)`
- `fuse_linear_relu2`: Replaces `F.relu(x)**2` with custom kernel
- Only active when corresponding kernel implementations exist

## Hardware Configuration (`hardware:`)

### Device and Precision
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `device` | cuda | cuda | Computation device |
| `enable_tf32` | true | true | Enable TensorFloat-32 (Ampere+) |
| `enable_bf16_reduced_precision` | true | true | Allow reduced precision BF16 ops |
| `cudnn_benchmark` | true | true | Enable CUDNN autotuner |
| `cudnn_deterministic` | false | false | Disable deterministic algorithms |

### Profiling and Scaling
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `profile` | a100_80gb_8x | - | Hardware profile identifier |
| `min_vram_gb` | 70.0 | - | Minimum VRAM required per GPU |
| `n_gpus` | 8 | - | Number of GPUs to utilize |
| `num_workers` | 8 | 4-16 | DataLoader worker processes |
| `val_num_workers` | 4 | 2-8 | Validation DataLoader workers |
| `prefetch_factor` | 4 | 2-8 | Batches to prefetch per worker |
| `empty_cache_every` | 0 | 0-1000 | Steps between empty_cache() calls (0=disabled) |
| `async_checkpointing` | true | true | Enable async checkpoint saving |
| `async_wandb` | true | true | Enable async W&B logging |
| `async_mlflow` | true | true | Enable async MLflow logging |
| `use_mmap_data` | true | true | Use memory-mapped data loading |
| `enable_nvlink_check` | true | true | Check NVLink availability at startup |

*Interaction*: 
- Settings primarily informational or used by setup routines
- `profile` used for logging/reporting only
- Memory settings help with OOM prevention
- Async features improve overlap of computation and I/O

## Model Configuration (`model:`)

### Architecture Basics
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `vocab_size` | 152064 | 152064 | Vocabulary size (Qwen2.5 BPE) |
| `max_seq_len` | 4096 | 2048-16384 | Maximum sequence length |
| `dim` | 2048 | 1024-4096 | Model dimension (hidden size) |
| `n_layers` | 30 | 16-64 | Number of Transformer blocks |
| `layer_schedule` | "5:1" | - | Layer type schedule (see below) |

*Interaction*: 
- `dim` must be divisible by `n_heads`
- `max_seq_len` affects memory usage quadratically for attention
- `n_layers` directly scales compute and memory

### Layer Schedule
Accepted formats for `layer_schedule`:
- `"mha"`: All layers use MLA attention
- `"ssm"`: All layers use GDN/Mamba-2 SSM
- `"ssm:N"`: Every N-th layer is SSM (e.g., `"ssm:6"` = every 6th layer)
- `"A:B"`: Repeating pattern of A MLA layers then B SSM layers (e.g., `"5:1"` = 5 MLA + 1 SSM)

*Examples*:
- `"5:1"`: 5 MLA blocks, 1 GDN block, repeat (6 layers per cycle)
- `"ssm:8"`: Every 8th layer is GDN, others MLA
- `"mha"`: All MLA (standard transformer)
- `"ssm"`: All GDN (pure SSM backbone)

### MLA Parameters (Multi-Head Latent Attention)
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `n_heads` | 32 | - | Number of query heads |
| `n_kv_groups` | 8 | - | Number of KV heads (GQA) |
| `q_lora_rank` | 512 | - | Query low-rank projection dimension |
| `kv_lora_rank` | 256 | - | Key/value low-rank projection dimension |
| `qk_nope_head_dim` | 128 | - | QK projection without RoPE |
| `qk_rope_head_dim` | 64 | - | QK projection with RoPE |
| `v_head_dim` | 128 | - | Value projection dimension |
| `rope_theta` | 10000.0 | - | RoPE base frequency |
| `rope_factor` | 8.0 | - | RoPE length scaling factor |
| `qk_norm` | true | true | Enable QK normalization |
| `sliding_window` | 2048 | - | Sliding window size for local attention (0=disabled) |
| `sliding_window_schedule` | "5:1" | - | Schedule for sliding window application |

*Interaction*: 
- Head dimension = `qk_nope_head_dim + qk_rope_head_dim` = 192
- Total QKV parameters reduced via low-rank factorization
- `n_kv_groups` must divide `n_heads` (here: 4 query heads per KV group)
- RoPE applied to `qk_rope_head_dim` dimensions per head
- Sliding window uses local attention over last `sliding_window` tokens

### Dense FFN (GDN Layers)
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `inter_dim` | 4096 | - | Intermediate dimension for dense FFN |
| `ffn_activation` | swiglu | swiglu\|relu2 | Activation function for dense FFN |

*Interaction*: 
- Used in GDN/SSM layers (every 6th layer with "5:1" schedule)
- SwiGLU: `W2(SiLU(W1(x)) * W3(x))` (3 weights)
- ReLU2: `W2(ReLU(W1(x)) ** 2)` (2 weights, legacy)
- Both variants use `bias=False`

### MoE FFN (MLA Layers)
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `n_routed_experts` | 64 | 32-128 | Number of routed experts |
| `n_shared_experts` | 4 | 2-8 | Number of always-active shared experts |
| `n_activated_experts` | 6 | 1-8 | Number of routed experts activated per token |
| `moe_inter_dim` | 1536 | - | Intermediate dimension per expert FFN |
| `expert_capacity_factor` | 1.5 | 1.0-2.0 | Expert capacity multiplier |
| `expert_dropout_prob` | 0.1 | 0.0-0.2 | Dropout probability on expert outputs |
| `moe_warmup_steps` | 2000 | 0-10000 | Steps for expert initialization warmup |
| `n_expert_groups` | 8 | 4-16 | Number of expert groups for group-limited routing |
| `n_limited_groups` | 3 | 1-4 | Number of groups selected per token |
| `group_topk` | 2 | 1-3 | Top experts selected per group |
| `route_scale` | 1.0 | - | Scaling factor for routed expert outputs |
| `bias_upper_threshold` | 0.10 | 0.05-0.2 | Upper threshold for bias updates |
| `bias_lower_threshold` | 0.10 | 0.05-0.2 | Lower threshold for bias updates |
| `moe_activation` | swiglu | swiglu\|relu2 | Activation function for MoE expert FFN |

*Interaction*: 
- Group-limited routing: tokens routed to `n_limited_groups` groups, then `group_topk` experts per group
- Total activated experts = `n_limited_groups × group_topk` = 3 × 2 = 6 with defaults
- Shared experts always contribute to FFN output
- Bias updates prevent expert collapse and improve load balancing
- Warmup period helps stabilize initial routing

### GDN (Gated Delta Net) Configuration
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `ssm_type` | gdn | gdn\|mamba2 | SSM variant selection |
| `gdn_d_state` | 128 | - | GDN state dimension |
| `gdn_d_conv` | 4 | - | GDN convolution width |
| `gdn_headdim` | 64 | - | GDN head dimension |

*Interaction*: 
- `ssm_type: gdn` uses Qwen3-Next Gated Delta Net
- `ssm_type: mamba2` uses legacy Mamba-2 selective scan
- Both implement the same `forward(x) -> y` interface
- GDN/Mamba-2 layers use dense SwiGLU FFN (not MoE)

### Mamba-2 Legacy Parameters
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `mamba_d_state` | 128 | - | Mamba-2 state dimension |
| `mamba_d_conv` | 4 | - | Mamba-2 convolution width |
| `mamba_headdim` | 64 | - | Mamba-2 head dimension |

*Interaction*: 
- Only used when `ssm_type: mamba2`
- Mirrors GDN parameters for easy switching

### Multi-Token Prediction (MTP)
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `mtp_depth` | 3 | 1-4 | Number of future tokens to predict |
| `mtp_loss_weight` | 0.30 | 0.1-0.5 | Weight of MTP loss relative to main loss |
| `mtp_loss_weight_schedule` | [0.3, 0.2, 0.1] | - | Per-depth loss weights (must sum to mtp_loss_weight) |
| `mtp_softcap` | true | true | Enable logit softcap for MTP heads |
| `mtp_softcap_value` | 15.0 | 10.0-30.0 | Softcap value for MTP logits |

*Interaction*: 
- MTP shares embedding and head weights with main model
- Loss weight schedule should have length = `mtp_depth`
- Each element corresponds to loss weight for predicting token t+1, t+2, etc.
- Auxiliary heads project from hidden state to vocab size

### μP (μ-transfer) Configuration
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `muP` | true | true\|false | Enable μP re-initialization |

*Interaction*: 
- When enabled, overrides standard initialization for certain parameters
- Aims to transfer hyperparameters from small to large models
- Implemented in `models/mup.py:muP_init()`
- Affects embedding, output head, and certain weight matrices

### Logit Softcap and Rescaling
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `logit_softcap` | 15.0 | 0.0-30.0 | Logit softcap value (0=disabled) |
| `asymmetric_rescale` | false | false\|true | Enable asymmetric rescaling layer |

*Interaction*: 
- Softcap: `logits = cap * tanh(logits / cap)`
- Prevents extreme logit values that cause instability
- Asymmetric rescaling: per-(channel,token) learnable affine transform
- Both applied after LM head, before cross-entropy loss

### Embedding and Head
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `tie_embeddings` | true | true | Tie input and output embedding weights |
| `no_bias_linear` | true | true | Disable bias in all linear layers (except MoE gate) |

*Interaction*: 
- Tied embeddings save ~2 × vocab_size × dim parameters
- `no_bias_linear` simplifies optimization and reduces parameters
- MoE gate biases are exempt from `no_bias_linear` (explicitly allowed)

## Data Configuration (`data:`)

### Paths
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `train_data_path` | data/pretrain_data.bin | - | Path to training token data |
| `val_data_path` | data/validation_data.bin | - | Path to validation token data |
| `shard_manifest_path` | None | - | Path to data shard manifest (for curriculum) |

*Interaction*: 
- Data files are expected to be torch.save'd tensors of token IDs
- When `shard_manifest_path` is set, uses `AsyncShardLoader`
- Otherwise uses standard `DataLoader` with `PretrainDataset`
- Manifest enables curriculum learning and dynamic data mixing

### Data Mix
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `fineweb_edu` | 0.60 | - | FineWeb-Edu weight |
| `finemath` | 0.15 | - | FineMath weight |
| `stack_edu` | 0.15 | - | Stack-Edu weight |
| `cosmopedia` | 0.05 | - | Cosmopedia weight |
| `openr1_math` | 0.05 | - | OpenR1-Math weight |

*Interaction*: 
- Weights should sum to 1.0
- Used by data preparation script (`data/prepare_data.py`)
- Only active when `shard_manifest_path` points to curriculum-aware manifest
- Static mixtures ignore these weights

### Tokenizer Settings
| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `vocab_size` | 152064 | - | Vocabulary size (must match model.vocab_size) |

*Interaction*: 
- Must match `model.vocab_size` for consistency
- Used for synthetic data generation when real data missing

## Configuration Interactions and Dependencies

### Critical Dependencies
1. **Model Parallelism**: 
   - `world_size` (from torchrun) must be compatible with FSDP strategy
   - `n_gpus` in hardware config should match actual GPU count

2. **Memory Constraints**:
   - `dim`, `n_layers`, `vocab_size` drive memory requirements
   - `use_checkpoint` reduces activation memory at compute cost
   - `fsdp_shard_strategy` affects memory communication tradeoffs

3. **Training Stability**:
   - `lr`, `muon_lr`, `min_lr_ratio` must be balanced
   - `weight_decay` and `cautious_wd` interaction affects convergence
   - `gradient_clip` prevents exploding gradients

4. **MoE Stability**:
   - `n_routed_experts` and `n_activated_experts` determine sparsity
   - `bias_update_speed` and `bias_update_every` affect load balancing
   - `expert_capacity_factor` prevents expert overflow

5. **Evaluation Tradeoffs**:
   - `eval_enabled` increases wall-clock time
   - `eval_synthetic: false` requires real data and lm_eval
   - `eval_interval` affects monitoring frequency vs training speed

### Recommended Configurations

#### Development / Debugging
```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  total_steps: 100
  warmup_steps: 10
  lr: 1e-4
  muon_lr: 0.001
  use_checkpoint: true
  eval_enabled: true
  eval_interval: 20
  eval_synthetic: true
  wandb_enabled: false
  mlflow_enabled: false
model:
  vocab_size: 4096
  max_seq_len: 128
  dim: 64
  n_layers: 2
  layer_schedule: "1:1"
  n_routed_experts: 2
  n_shared_experts: 1
  n_activated_experts: 1
```

#### Training Run (Canonical 8×A100 SXM 80GB)
As defined in `configs/pretrain.yaml` - suitable for 150B token training.

#### Ablation Studies
To test specific components:
- **Disable MoE**: Set `n_routed_experts: 0` (falls back to dense FFN)
- **Pure MLA**: Set `layer_schedule: "mha"`
- **Pure SSM**: Set `layer_schedule: "ssm"`
- **Disable MTP**: Set `mtp_depth: 0`
- **Disable μP**: Set `muP: false`
- **Disable Softcap**: Set `logit_softcap: 0.0`

## Configuration Validation
The configuration system performs basic validation:
- Required fields are checked in `build_config_from_yaml()`
- Some ranges validated implicitly (e.g., positive integers)
- FSDP strategy validated in `utils/distributed.py:wrap_fsdp2()`
- Scheduler types validated in `training/pretrain.py`

Invalid configurations typically result in clear error messages during initialization.

## Sources
- Primary definitions: `training/pretrain.py:ConfigBundle` and nested dataclasses
- Smoke test config: `configs/smoke_pretrain.yaml`
- Main training config: `configs/pretrain.yaml`
- FSDP2 handling: `utils/distributed.py`
- Optimizer building: `training/pretrain.py:_build_optimizers()`
- Scheduler selection: `training/pretrain.py`