# Glossary

## Repository-Specific Terminology

### MLA (Multi-Head Latent Attention)
An attention mechanism that uses low-rank projections to compress the key and value matrices, significantly reducing memory bandwidth and storage requirements for the KV cache. Implemented in `models/mla.py`.

### GDN (Gated Delta Net)
A state space model (SSM) variant based on Qwen3-Next, used as an alternative to traditional attention. Implemented in `models/gated_deltanet.py`. Can be replaced with legacy Mamba-2 via `ssm_type: "mamba2"`.

### MoE (Mixture-of-Experts)
A technique where different expert networks process different parts of the input, activated via a routing mechanism. This implementation uses DeepSeekMoE with group-limited routing and bias updates. Located in `models/moe.py`.

### MTP (Multi-Token Prediction)
An auxiliary training objective where the model predicts multiple future tokens simultaneously, improving reasoning capabilities. Implemented in `models/mtp.py` with depth=3 by default.

### μP (μ-transfer)
A parameterization method that enables transferring hyperparameters from small to large scale models without retraining. Implemented in `models/mup.py`.

### SSM (State Space Model)
A general class of models that map sequences to sequences using a latent state space. Includes GDN and Mamba-2 implementations in this codebase.

### FFN (Feed-Forward Network)
The position-wise fully connected network in transformer blocks. Comes in two variants:
- Dense FFN: Used in GDN/SSM layers (`models/transformer.py:DenseFFN`)
- MoE FFN: Used in MLA layers (`models/moe.py:DeepSeekMoE`)

### FSDP2 (Fully Sharded Data Parallel 2)
The latest version of PyTorch's distributed data parallelism that shards parameters, gradients, and optimizer states across data parallel workers. Implemented via `torch.distributed.fsdp.fully_shard` in `utils/distributed.py`.

### Activation Checkpointing
A technique that trades compute for memory by discarding certain activations during forward pass and recomputing them during backward pass. Applied per TransformerBlock when `use_checkpoint: true`.

### CautiousAdamW
A variant of AdamW that applies weight decay only when the gradient and parameter have the same sign, preventing divergence in certain scenarios. Implemented in `training/pretrain.py:CautiousAdamW`.

### Muon / NorMuon
Optimizers based on Newton-Schulz orthogonalization for matrix parameters. Muon uses momentum, NorMuon uses Adam-style moment statistics. Used for weight matrices with ndim ≥ 2 (excluding embed/head).

### WSD (Warmup-Stable-Decay)
A learning rate scheduler with three phases: linear warmup, stable constant LR, and decay to minimum LR. Implemented in `training/wsd.py`.

## Acronyms

| Acronym | Full Form | Description |
|---------|-----------|-------------|
| MLA | Multi-Head Latent Attention | Low-rank compressed attention mechanism |
| GDN | Gated Delta Net | Qwen3-Next style state space model |
| MoE | Mixture-of-Experts | Sparse expert-based feed-forward network |
| MTP | Multi-Token Prediction | Predicting multiple future tokens |
| μP | Mu-transfer | Parameterization for stable scaling |
| SSM | State Space Model | General sequence modeling framework |
| FFN | Feed-Forward Network | Position-wise fully connected layer |
| FSDP | Fully Sharded Data Parallel | PyTorch distributed sharding strategy |
| activation checkpointing | Gradient checkpointing | Memory optimization technique |
| LoRA | Low-Rank Adaptation | Matrix factorization technique |
| GQA | Grouped Query Attention | Multiple queries sharing key/value heads |
| RoPE | Rotary Position Embedding | Rotary embedding for position information |
| SwiGLU | Sigmoid-weighted Gated Linear Unit | Activation function: silu(x1) * x2 |
| ReLU² | Squared ReLU | Activation function: relu(x)² |
| BF16 | Brain Floating Point 16-bit | 16-bit floating point format |
| FP16 | Floating Point 16-bit | 16-bit floating point format |
| FP8 | Floating Point 8-bit | 8-bit floating point format (E4M3) |
| TF32 | TensorFloat-32 | 19-bit format for tensor cores |
| DCP | Distributed Checkpointing | PyTorch's distributed checkpoint format |
| WSD | Warmup-Stable-Decay | Learning rate scheduler |
| LR | Learning Rate | Optimization step size |
| VRAM | Video Random Access Memory | GPU memory |
| HBM | High Bandwidth Memory | GPU memory technology |
| SM | Streaming Multiprocessor | GPU compute unit |
| NCCL | NVIDIA Collective Communications Library | GPU communication library |
| NVLink | NVIDIA's high-speed interconnect | GPU-GPU communication |
| PCIe | Peripheral Component Interconnect Express | Standard GPU interconnection |
| OOM | Out Of Memory | Memory exhaustion condition |
| NaN | Not a Number | Undefined floating-point value |
| Inf | Infinity | Infinite floating-point value |

## Custom Modules

| Module | Location | Purpose |
|--------|----------|---------|
| ParallelEmbedding | models/transformer.py | Vocab-sharded embedding with all-reduce |
| TransformerBlock | models/transformer.py | One transformer block with MLA/GDN option |
| Transformer | models/transformer.py | Full model backbone |
| MultiHeadLatentAttention | models/mla.py | MLA implementation |
| DeepSeekMoE | models/moe.py | Mixture-of-experts FFN |
| GatedDeltaNet | models/gated_deltanet.py | GDN SSM implementation |
| Mamba2Block | models/mamba.py | Mamba-2 SSM implementation |
| MultiTokenPrediction | models/mtp.py | Auxiliary heads for future token prediction |
| MuPInit | models/mup.py | μ-transfer re-initialization |
| NorMuon | training/normuon.py | Newton-Schulz + Adam optimizer |
| Muon | training/pretrain.py | Newton-Schulz + momentum optimizer |
| CautiousAdamW | training/pretrain.py | Sign-masked AdamW |
| WarmupCosineDecayScheduler | training/pretrain.py | Standard LR scheduler |
| WSDScheduler | training/wsd.py | Warmup-Stable-Decay scheduler |
| BatchSizeSchedule | training/schedules.py | Batch size scheduling |
| SeqLenSchedule | training/schedules.py | Sequence length scheduling |
| AsyncShardLoader | data/async_loader.py | Async sharded data loader |
| Curriculum | data/curriculum.py | Curriculum learning |
| CheckpointManager | utils/checkpoint.py | Model checkpointing |
| TrainerLogger | utils/logging.py | W&B/MLflow/CSV logging |
| Fused CE + Softcap | kernels/ce_softcap.py | Fused loss kernel |
| Fused Linear + ReLU² | kernels/linear_relu2.py | Fused activation kernel |
| FP8 Matmul | kernels/fp8_mla.py | FP8 matrix multiplication |
| Grouped GEMM | ops/triton/grouped_gemm.py | Triton expert computation |

## Research References

The implementation appears to draw inspiration from the following research (based on architectural choices and comments):

1. **DeepSeek-V2 / DeepSeek-V3** 
   - Multi-Head Latent Attention (MLA)
   - DeepSeekMoE with auxiliary-loss-free bias updates
   - Multi-Token Prediction (MTP)
   - Referenced in model comments and configuration

2. **Qwen3-Next**
   - Gated Delta Net (GDN) as SSM alternative
   - Referenced in `configs/pretrain.yaml` comments and model docs

3. **Mamba-2 / Mamba**
   - State space model alternative to attention
   - Referenced as legacy option (`ssm_type: "mamba2"`)

4. **Nemotron-H / Jamba**
   - Hybrid attention/SSM schedules (6:1 and 8:1 patterns)
   - Referenced in `models/transformer.py` schedule documentation

5. **Keller Jordan (Muon)**
   - Newton-Schulz orthogonalized momentum optimizer
   - Referenced in `training/normuon.py` and `training/pretrain.py`

6. **μTransfer (μP)**
   - Parameter transfer from small to large models
   - Referenced in `models/mup.py` and training config

7. **FlashAttention / FlashAttention-2 / FlashAttention-3**
   - Fast attention implementation
   - Referenced as optional (`use_fa3: false`)

8. **Rotary Position Embedding (RoPE)**
   - Position encoding with rotary transformations
   - Implemented in `models/rope.py`

9. **Grouped Query Attention (GQA)**
   - Multiple queries sharing key/value heads
   - Implemented via `n_kv_groups` in MLA

10. **SwiGLU Activation Function**
    - Gated linear unit with sigmoid weighting
    - Used in FFN implementations

11. **BF16 Training**
    - Brain floating point 16-bit precision training
    - Standard precision setting (`dtype: bf16`)

12. **Gradient Checkpointing**
    - Activation recomputation to save memory
    - Enabled by default (`use_checkpoint: true`)

13. **FSDP2 (Fully Sharded Data Parallel 2)**
    - Latest PyTorch distributed sharding
    - Core distributed strategy (`fsdp_shard_strategy: FULL_SHARD`)

14. **Warmup-Stable-Decay (WSD) Scheduler**
    - Three-phase learning rate schedule
    - Default scheduler (`scheduler: wsd`)

15. **Curriculum Learning**
    - Progressive difficulty or data mixing strategies
    - Implemented in `data/curriculum.py`

16. **TensorFloat-32 (TF32)**
    - 19-bit format for tensor core operations
    - Enabled by default (`enable_tf32: true`)

17. **NVLink Utilization**
    - High-speed GPU interconnect for communication
    - Checked at startup (`enable_nvlink_check: true`)

Note: While these references appear to inform the implementation, the codebase represents a unique combination and extension of these ideas rather than a direct copy of any single paper.