# Decision Log

This file documents architectural decisions inferred from the codebase, explaining why major components exist and what tradeoffs were accepted.

## Architectural Decisions

### 1. Hybrid MLA + GDN Architecture
**Decision**: Use a hybrid architecture alternating Multi-Head Latent Attention (MLA) and Gated Delta Net (GDN) layers instead of pure transformer or pure SSM.

**Evidence**:
- `models/transformer.py`: Implements both MLA and GDN paths
- `configs/pretrain.yaml`: `layer_schedule: "5:1"` (5 MLA + 1 GDN)
- Comments reference Nemotron-H (6:1) and Jamba (8:1) patterns

**Why**:
- **Quality/Efficiency Tradeoff**: MLA provides high-quality attention for critical layers, GDN provides efficient inference for others
- **Inference Speed**: GDN layers offer constant-time inference vs quadratic for attention
- **Memory Savings**: GDN has smaller memory footprint than attention layers
- **Best of Both Worlds**: Combines strengths of attention (expressivity) and SSM (efficiency)

**Tradeoffs Accepted**:
- Increased implementation complexity
- Need to tune layer schedule ratio
- Potential quality drop on tasks requiring pure attention
- More complex debugging due to two different layer types

### 2. Multi-Head Latent Attention (MLA)
**Decision**: Implement MLA with low-rank KV compression, decoupled RoPE, and GQA instead of standard multi-head attention.

**Evidence**:
- `models/mla.py`: Full MLA implementation
- Config parameters: `q_lora_rank: 512`, `kv_lora_rank: 256`, `n_kv_groups: 8`
- Comments describing low-rank KV cache, GQA-on-top, optional sliding window

**Why**:
- **Memory Bandwidth Reduction**: Low-rank projections reduce KV cache size significantly
- **Computational Efficiency**: Smaller matrix operations in attention computation
- **Quality Preservation**: Decoupled RoPE maintains positional information
- **Scalability**: Enables longer context lengths with same memory budget

**Tradeoffs Accepted**:
- Approximation error from low-rank compression
- Additional implementation complexity
- Need to tune compression ranks (`q_lora_rank`, `kv_lora_rank`)
- Potential quality loss on certain tasks requiring full attention

### 3. DeepSeekMoE with Group-Limited Routing
**Decision**: Use fine-grained Mixture-of-Experts with group-limited routing, bias updates, and shared experts instead of standard MoE or dense FFN.

**Evidence**:
- `models/moe.py`: DeepSeekMoE implementation
- Config: `n_routed_experts: 64`, `n_activated_experts: 6`, `n_shared_experts: 4`
- Group-limited routing: `n_expert_groups: 8`, `n_limited_groups: 3`, `group_topk: 2`
- Bias update mechanism: `bias_update_speed: 1e-3`, `bias_update_every: 10`

**Why**:
- **Computational Efficiency**: Only ~15.6% of FFN parameters active per token
- **Capacity Scaling**: Massive parameter count with constant compute
- **Load Balancing**: Bias updates prevent expert collapse
- **Expert Specialization**: Grouping encourages expert specialization
- **Shared Knowledge**: Shared experts capture common patterns

**Tradeoffs Accepted**:
- Routing overhead and complexity
- Potential expert imbalance despite mechanisms
- Implementation complexity of group-limited routing
- Need for bias update tuning
- Communication overhead in distributed settings

### 4. Gated Delta Net (GDN) as Default SSM
**Decision**: Use Qwen3-Next Gated Delta Net as the default SSM implementation, with legacy Mamba-2 as opt-in.

**Evidence**:
- `models/gated_deltanet.py`: GDN implementation
- `models/mamba.py`: Mamba-2 implementation (legacy)
- Config: `ssm_type: "gdn"` (default) with comment about legacy option
- Comments in transformer.py describing GDN as Qwen3-Next style

**Why**:
- **Performance**: GDN reported to have better performance than Mamba-2
- **Stability**: Possibly more stable training dynamics
- **Modern Design**: Based on more recent Qwen3-Next architecture
- **Configurability**: Easy switching via `ssm_type` parameter

**Tradeoffs Accepted**:
- Maintaining two similar implementations increases code complexity
- Need to validate both options work correctly
- Potential confusion about which to use
- Legacy option may receive less testing

### 5. MuT (Multi-Token Prediction)
**Decision**: Implement MTP with depth=3 to predict tokens 1, 2, and 3 steps ahead.

**Evidence**:
- `models/mtp.py`: MTP implementation
- Config: `mtp_depth: 3`, `mtp_loss_weight: 0.30`
- Comments describing MTP depth and loss weights
- Integration in `training/pretrain.py` with MTP wrapper

**Why**:
- **Reasoning Improvement**: Predicting multiple steps improves logical reasoning
- **Data Efficiency**: Uses same forward pass for multiple prediction targets
- **Consistency**: Encourages consistent representations across time steps
- **Auxiliary Loss**: Provides additional training signal without much overhead

**Tradeoffs Accepted**:
- Additional computation for auxiliary heads
- Complexity in loss weighting and balancing
- Potential interference with main language modeling objective
- Need to tune MTP depth and loss weights

### 6. μP (μ-transfer) Re-initialization
**Decision**: Enable μP re-initialization to transfer hyperparameters from small to large scale models.

**Evidence**:
- `models/mup.py`: μP implementation
- Config: `muP: true`
- Comments in transformer.py about μP re-initialization
- Call to `muP_init()` in Transformer constructor

**Why**:
- **Stable Scaling**: Allows training large models with settings from small models
- **Reduced Tuning**: Less extensive hyperparameter search at large scale
- **Theoretical Grounding**: Based on sound mathematical principles
- **Empirical Validation**: Presumably validated in research

**Tradeoffs Accepted**:
- Implementation complexity
- Need to verify correctness of μP transformations
- Potential interactions with other initialization schemes
- May not transfer perfectly in all cases

### 7. Dual Optimizer Strategy (Muon/NorMuon + CautiousAdamW)
**Decision**: Use separate optimizers for matrix parameters (Muon/NorMuon) and others (CautiousAdamW).

**Evidence**:
- `training/pretrain.py:_build_optimizers()`: Parameter separation logic
- `training/normuon.py`: NorMuon implementation
- References to Muon in comments and code
- Config: `optimizer: normuon_adamw`

**Why**:
- **Different Optimization Needs**: Matrix weights (weights) vs embeddings/biases have different properties
- **Muon Benefits**: Newton-Schulz orthogonalization prevents rank collapse in weight matrices
- **CautiousAdamW Benefits**: Sign-masked weight decay prevents divergence in embedding/biases
- **Empirical Success**: Based on Keller Jordan's NanoGPT work showing improved training stability

**Tradeoffs Accepted**:
- Implementation complexity of maintaining two optimizers
- Need to verify parameter separation logic is correct
- Potential tuning complexity with two learning rates
- Slightly more overhead in optimizer stepping

### 8. WSD (Warmup-Stable-Decay) Scheduler
**Decision**: Use WSD learning rate scheduler with linear warmup, stable period, and decay.

**Evidence**:
- `training/wsd.py`: WSD scheduler implementation
- Config: `scheduler: wsd`, `wsd_warmup_frac: 0.01`, `wsd_stable_frac: 0.84`
- Fallback to cosine scheduler in code

**Why**:
- **Stable Training**: Extended stable period at max LR helps convergence
- **Prevents Over-decay**: Avoids decaying too early in training
- **Flexible**: Can use linear or cosine decay to minimum LR
- **Empirical Validation**: Based on research showing benefits for large batch training

**Tradeoffs Accepted**:
- More complex than simple cosine or linear decay
- Three hypermparameters to tune (warmup, stable, decay type)
- May not be optimal for all architectures or datasets
- Slightly more implementation complexity

### 9. FSDP2 with Per-Block Wrapping
**Decision**: Use FSDP2 with per-TransformerBlock wrapping instead of wrapping the entire model or using finer/coarser granularity.

**Evidence**:
- `utils/distributed.py:wrap_fsdp2()`: Per-TransformerBlock wrapping logic
- Comments explaining the wrapping policy choice
- Usage in `training/pretrain.py` during model initialization

**Why**:
- **Good Granularity**: Blocks are natural units for sharding
- **Expert Inclusion**: Experts get sharded alongside their blocks
- **Memory Efficiency**: Better than wrapping entire model (less padding)
- **Communication Efficiency**: Better than finer granularity (less all-gather overhead)
- **Standard Practice**: Common approach for transformer-based models

**Tradeoffs Accepted**:
- Slightly less optimal than perfect per-parameter sharding
- Need to identify correct block class for wrapping
- Potential load imbalance if blocks have very different sizes
- Expert parallelism not explicitly utilized (relying on FSDP sharding)

### 10. Activation Checkpointing per TransformerBlock
**Decision**: Apply gradient checkpointing at the TransformerBlock level.

**Evidence**:
- `training/pretrain.py`: `use_checkpoint: true` in config
- `models/transformer.py:TransformerBlock.forward()` wrapped with `torch.utils.checkpoint.checkpoint`
- Comments about gradient checkpointing per block

**Why**:
- **Significant Memory Savings**: Reduces activation memory from O(layers) to O(√layers)
- **Reasonable Compute Cost**: ~33% increase in compute time
- **Natural Granularity**: Blocks are self-contained with clear inputs/outputs
- **Wide Applicability**: Works with most standard transformer operations

**Tradeoffs Accepted**:
- ~33% increase in compute time (forward recomputation)
- Slightly more complex debugging due to recomputation
- May not save as much memory as finer-grained checkpointing
- Some operations may not be compatible with checkpointing

### 11. Tied Input/Output Embeddings
**Decision**: Tie the input embedding and output LM head weights to reduce parameters.

**Evidence**:
- `models/transformer.py`: `self.head.weight = embed.weight` when `tie_embeddings: true`
- Config: `tie_embeddings: true`
- Comments about parameter savings from tying

**Why**:
- **Parameter Efficiency**: Saves ~2 × vocab_size × dim parameters
- **No Quality Loss**: Empirically shown to not hurt performance
- **Consistency**: Input and output representations should be related
- **Standard Practice**: Used in Llama, Qwen, Phi models

**Tradeoffs Accepted**:
- None significant; pure benefit with no apparent downsides
- Must ensure embedding dimension matches hidden dimension
- Output bias must be handled separately (typically no bias)

### 12. Bias-Free Linear Layers (Except MoE Gate)
**Decision**: Disable bias in all linear layers except the MoE gate.

**Evidence**:
- `models/transformer.py`: `no_bias_linear: true` in config
- Comments: "no bias in any linear (except MoE gate)"
- Checking implementations confirms no bias in most linears

**Why**:
- **Parameter Efficiency**: Eliminates unnecessary bias parameters
- **Optimization Simplicity**: Fewer parameters to optimize
- **Empirical Validation**: Works well in practice (Llama, etc.)
- **MoE Exception**: Gate biases are important for routing dynamics

**Tradeoffs Accepted**:
- Very minimal; biases often not critical in deep networks
- Must verify MoE gate biases are correctly exempted
- Possible slight impact on optimization landscape

### 13. BF16 Precision with FP32 Reduction
**Decision**: Use BF16 for most computations but FP32 for precision-sensitive operations.

**Evidence**:
- `configs/pretrain.yaml`: `dtype: bf16`
- `utils/distributed.py`: `fsdp_reduce_dtype: fp32` for gradient reductions
- Model code: Likely uses autocast for BF16 with FP32 reduction in softmax/norms

**Why**:
- **Memory Bandwidth**: BF16 reduces memory traffic by 2x vs FP32
- **Computational Throughput**: Better utilization of tensor cores
- **Numerical Stability**: FP32 reduction prevents precision loss in reductions
- **Industry Standard**: Used in most modern LLM training

**Tradeoffs Accepted**:
- Slight precision reduction in BF16 operations
- Need to identify which operations require FP32
- Potential complexity in mixed precision management
- Very small quality impact typically observed

### 14. Gradient Clipping
**Decision**: Apply gradient clipping with norm=1.0.

**Evidence**:
- `configs/pretrain.yaml`: `grad_clip: 1.0`
- `training/pretrain.py`: `nn.utils.clip_grad_norm_(params, self.cfg.optim.max_grad_norm)`

**Why**:
- **Training Stability**: Prevents exploding gradients
- **Robustness**: Helps training recover from instability spikes
- **Standard Practice**: Universally used in deep learning training
- **Low Cost**: Minimal computational overhead

**Tradeoffs Accepted**:
- May prevent legitimate large gradients in rare cases
- Requires tuning of clipping threshold
- Can mask underlying instability issues if overused
- Very small impact on final convergence typically

### 15. Async Checkpointing and Logging
**Decision**: Enable asynchronous checkpointing and logging to overlap with computation.

**Evidence**:
- `configs/pretrain.yaml`: `async_checkpointing: true`, `async_wandb: true`, `async_mlflow: true`
- `utils/checkpoint.py` and `utils/logging.py`: Likely use background threads/processes
- Comments about async options being performance optimizations

**Why**:
- **Training Efficiency**: Overlaps I/O with computation
- **Reduced Stall Time**: Less time spent waiting for disk/network
- **Better Utilization**: Keeps GPUs busy during save/log operations
- **Scalability**: More important as checkpoint frequency increases

**Tradeoffs Accepted**:
- Increased complexity in error handling
- Potential for lost logs/checkpoints if process crashes mid-async
- Slightly higher memory usage for buffering
- Need to ensure thread/process safety

### 16. Curriculum Learning Framework
**Decision**: Implement a two-stage curriculum learning system for data mixing.

**Evidence**:
- `data/curriculum.py`: Curriculum implementation
- Config: `curriculum_switch_step: 0` (disabled by default)
- Comments describing stage-1 (web-heavy) and stage-2 (code/math-heavy) weights

**Why**:
- **Progressive Learning**: Start with easier/web data, move to specialized data
- **Data Efficiency**: Focus compute on most relevant data at each stage
- **Mitigates Forgetting**: Helps prevent forgetting earlier learned concepts
- **Flexibility**: Easy to enable/disable and configure

**Tradeoffs Accepted**:
- Implementation complexity
- Need to prepare data manifest and weights
- Risk of switching too early/late
- May not benefit all types of data or models
- Requires careful weighting to avoid data starvation

### 17. Synthetic Data Fallback
**Decision**: Generate random token data when real data is missing.

**Evidence**:
- `training/pretrain.py:PretrainDataset`: Generates synthetic data if file missing
- `eval/eval_core.py:make_synthetic_loader()`: Deterministic synthetic loader
- Comments about synthetic data for testing

**Why**:
- **Developer Experience**: Enables testing without real data setup
- **CI/CD Friendly**: Allows automated testing in clean environments
- **Debugging**: Provides predictable data for reproducing issues
- **Fallback Safety**: Prevents training failures due to missing data

**Tradeoffs Accepted**:
- Synthetic data doesn't represent real language statistics
- May mask data pipeline issues
- Not suitable for actual training runs
- Potential confusion if not clearly marked as synthetic

### 18. Deterministic Synthetic Loader for Evaluation
**Decision**: Use deterministic random data for evaluation when real validation data unavailable.

**Evidence**:
- `eval/eval_core.py:make_synthetic_loader()`: Uses fixed seed for reproducibility
- `training/pretrain.py:_maybe_eval()`: Uses this when `eval_synthetic: true`

**Why**:
- **Reproducibility**: Same evaluation scores across runs
- **Debugging**: Consistent baseline for measuring changes
- **CI Testing**: Reliable for automated testing
- **Resource Efficiency**: No need for large validation datasets

**Tradeoffs Accepted**:
- Doesn't measure real-world performance
- May overfit to peculiarities of synthetic distribution
- Not suitable for final performance reporting
- Must clearly distinguish from real evaluation

### 19. Logit Softcap
**Decision**: Apply logit softcap with value=15.0 to prevent extreme logit values.

**Evidence**:
- `configs/pretrain.yaml`: `logit_softcap: 15.0`
- `models/transformer.py`: `_logit_cap` and `softcap_15()` function
- Comments about DeepSeek-V3 logit soft-cap

**Why**:
- **Training Stability**: Prevents extremely large logits causing instability
- **Numerical Safety**: Bounded logits reduce overflow/underflow risk
- **Calibration**: Helps prevent overconfident predictions
- **DeepSeek-V3 Practice**: Following established recipe from their work

**Tradeoffs Accepted":
- May slightly limit model expressiveness
- Requires tuning of cap value
- Small computational overhead
- Potential underfitting if cap too low

### 20. Sliding Window Attention with Local-Global Interleaving
**Decision**: Use sliding window attention with 5:1 local-global interleaving pattern.

**Evidence**:
- `models/mla.py`: Sliding window implementation
- Config: `sliding_window: 2048`, `sliding_window_schedule: "5:1"`
- Comments about Gemma 2 5:1 local-global interleaving

**Why**:
- **Memory Efficiency**: Sliding window reduces KV cache from O(seq_len) to O(window)
- **Local Context**: Captures immediate context efficiently
- **Global Context**: Periodic global attention captures long-range dependencies
- **Gemma 2 Inspiration**: Following proven architectural pattern

**Tradeoffs Accepted":
- Implementation complexity for schedule handling
- Need to tune window size and schedule ratio
- May lose some very long-range dependencies
- Extra computation for schedule checking

## Summary of Key Tradeoffs Patterns

Across these decisions, several patterns emerge in the tradeoffs accepted:

1. **Complexity for Efficiency**: Most decisions increase implementation complexity to gain memory or computational efficiency (MLA, MoE, GDN, activation checkpointing, tied embeddings).

2. **Quality/Efficiency Balance**: Hybrid architecture, sliding window, and MTP all balance quality and efficiency concerns.

3. **Stability Over Peak Performance**: Choices like cautious weight decay, gradient clipping, logit softcap, and μP prioritize training stability over absolute peak performance.

4. **Empirical Validation**: Many decisions follow established practices from successful models (Llama, Qwen, DeepSeek, Nemotron-H/Jamba).

5. **Scalability Focus**: Decisions consistently prioritize ability to scale to larger models and longer sequences over minimal implementation.

6. **Configuration-Driven Flexibility**: Most major architectural choices are configurable via YAML, enabling experimentation without code changes.

These decisions collectively reflect a mature, research-informed approach to building efficient, scalable language models that prioritize practical training stability and scalability while maintaining competitive performance.