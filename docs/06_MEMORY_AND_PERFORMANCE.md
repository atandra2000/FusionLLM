# Memory and Performance Analysis

## VRAM Analysis
Based on the canonical configuration (8×A100 SXM 80GB) and model specifications:

### Memory Breakdown per GPU
From `configs/pretrain.yaml` comments:
- **Static State**: ~4.5 GB/GPU (model parameters, optimizer states, gradients)
- **Activation Memory**: Dynamically allocated; managed via gradient checkpointing
- **KV Cache**: Depends on sequence length and batch size
- **Headroom**: Activations + KV cache should fit comfortably in remaining VRAM (~68.5 GB)

### Detailed Component Estimates
#### Model Parameters
- **Total Parameters**: ~7B (as noted in config)
- **Storage**: 
  - FP32: 28 GB
  - BF16: 14 GB 
  - With FSDP2 FULL_SHARD (8-way): ~1.75 GB/GPU for parameters

#### Optimizer States (AdamW-style)
- **Per Parameter**: 2x (momentum + variance) for AdamW components
- **NorMuon/Muon**: Additional states for orthogonalization momentum
- **Storage**: ~2x parameter storage for optimizer states
- **With FSDP2**: ~3.5 GB/GPU for optimizer states

#### Gradients
- **Storage**: Same as parameters (BF16)
- **With FSDP2**: ~1.75 GB/GPU for gradients

#### Activation Memory (with Checkpointing)
- **Without Checkpointing**: O(layers × batch × seq_len × dim)
- **With Checkpointing**: O(√layers × batch × seq_len × dim) approximately
- **Estimate**: For batch=2, seq_len=4096, dim=2048, n_layers=30:
  - Full activation: ~30 × 2 × 4096 × 2048 × 2 bytes ≈ 1 GB
  - With checkpointing: Reduced by factor of ~√30 ≈ 5.5 → ~180 MB

#### KV Cache
- **Storage**: 2 × batch × seq_len × n_kv_heads × head_dim × sizeof(dtype)
- **MLA KV**: Compressed via low-rank (kv_lora_rank=256)
- **Effective KV Size**: Much smaller than standard attention
- **Estimate**: For batch=2, seq_len=4096:
  - Standard: 2 × 4096 × 32 × 128 × 2 bytes ≈ 128 MB
  - MLA compressed: Significantly less due to low-rank projection

### Total Estimated Memory per GPU
- Parameters: ~1.75 GB
- Optimizer States: ~3.5 GB  
- Gradients: ~1.75 GB
- Activations: ~0.2 GB (with checkpointing)
- KV Cache: ~0.1 GB (estimated)
- **Total**: ~7.3 GB
- **Headroom**: ~24.7 GB for other overheads, memory fragmentation, etc.

This aligns with the config comment: "static state ≈ 4.5 GB/GPU" (likely includes some activation overhead in their measurement).

## Activation Memory
### Gradient Checkpointing Implementation
- **Location**: `models/transformer.py:TransformerBlock._forward()` wrapped with `torch.utils.checkpoint.checkpoint`
- **Control**: `use_checkpoint: true` in training config
- **Granularity**: Per TransformerBlock (activation recomputation between residuals)
- **Trade-off**: Saves ~30-40% activation memory at cost of ~33% extra compute (forward recompute)

### Activation Components
1. **Input Embeddings**: `[batch, seq_len, dim]`
2. **Attention/SSM Output**: `[batch, seq_len, dim]` 
3. **FFN Intermediate**: `[batch, seq_len, inter_dim]` (SwiGLU splits)
4. **Residual Connections**: Multiple `[batch, seq_len, dim]` tensors
5. **Norm Statistics**: Mean/variance for RMSNorm (minimal)

### Memory Optimization Techniques
- **In-place Operations**: Where safe (e.g., `x = x + attn(...)`)
- **Tensor Reuse**: Minimizing temporary allocations
- **Efficient Kernels**: Custom CUDA kernels reduce memory traffic

## MLA Savings
### Standard Multi-Head Attention Memory
For comparison, standard MHA would require:
- **QKV Projections**: 3 × `[batch, seq_len, dim, dim]` weight matrices
- **Attention Scores**: `[batch, n_heads, seq_len, seq_len]` (major bottleneck)
- **Context Output**: `[batch, seq_len, dim]`

### MLA (Multi-Head Latent Attention) Improvements
#### Parameter Reduction
- **Standard MHA QKV**: 3 × dim² = 3 × 2048² ≈ 25M parameters
- **MLA QKV**: 
  - Query: dim → q_lora_rank → (qk_nope_head_dim + qk_rope_head_dim) × n_heads
  - KV: dim → kv_lora_rank → (qk_nope_head_dim + qk_rope_head_dim) × n_kv_groups
  - Value: dim → kv_lora_rank → v_head_dim × n_kv_groups
  - Output: (v_head_dim × n_kv_groups) × dim
- **Estimated Reduction**: ~5-10x fewer QKV parameters

#### KV Cache Reduction
- **Standard MHA KV Cache**: 2 × batch × seq_len × n_heads × head_dim
- **MLA KV Cache**: 2 × batch × seq_len × kv_lora_rank × (qk_nope_head_dim + qk_rope_head_dim) × (n_kv_groups / n_heads)
- **With Values**: 
  - Standard: 2 × 2 × 4096 × 32 × 128 × 2 bytes ≈ 256 MB
  - MLA: Much smaller due to low-rank projection (kv_lora_rank=256 vs expected ~4096)

### Computational Benefits
- **Reduced Memory Bandwidth**: Smaller weight matrices to load
- **Faster Kernel Launch**: Smaller matrix dimensions
- **Better Cache Utilization**: More operations per loaded weight

## MoE Memory Behavior
### Activation Memory
- **Routing**: Gate computation produces `[batch, seq_len, n_routed_experts]` scores
- **Top-k Selection**: Indices and weights for activated experts
- **Expert Computation**: Each token routed to `n_activated_experts` experts
- **Memory Pattern**: Sparse computation reduces active memory footprint

### Parameter Storage
- **Expert Weights**: `n_routed_experts` × FFN parameters per expert
- **With FSDP2**: Sharded alongside other block parameters
- **Active Expert Fraction**: `(n_activated_experts + n_shared_experts) / n_routed_experts`
- **With Defaults**: (6 + 4) / 64 = 15.6% of expert FFN parameters active per token

### Memory Savings vs Dense FFN
- **Standard FFN**: `dim → inter_dim → dim` parameters
- **MoE FFN**: Same per expert, but only fraction active
- **Effective Memory Reduction**: ~84% fewer expert parameters accessed per forward pass
- **Trade-off**: Routing overhead and imbalance management

### Memory Characteristics During Training
1. **Forward Pass**:
   - Gate scores: `[batch, seq_len, n_routed_experts]` (BF16)
   - Routing indices/weights: Small int/float tensors
   - Expert outputs: Only activated experts computed
   - Combine: Weighted sum of expert outputs

2. **Backward Pass**:
   - Gradients flow back through activated experts only
   - Expert gradients accumulated for optimizer step
   - Gate bias updates: Lightweight statistics

## Checkpointing Strategy
### Activation Checkpointing (Gradient Checkpointing)
- **Implementation**: `torch.utils.checkpoint.checkpoint` in `TransformerBlock._forward()`
- **Scope**: Recomputes activations between `norm1(x)` and `norm2(x)` outputs
- **Frequency**: Applied to every TransformerBlock when `use_checkpoint: true`
- **Benefit**: Reduces activation memory from O(layers) to O(√layers) approximately
- **Cost**: ~33% increase in compute time (forward recomputation during backward)

### Model Checkpointing
- **Implementation**: `utils/checkpoint.py:CheckpointManager`
- **Triggers**: 
  - Interval-based: `save_interval` steps (default 500)
  - Manual: `save_checkpoint()` calls
  - Final: Tagged as "final"
- **Frequency Control**: 
  - Training: `save_interval: 500` steps
  - Smoke test: `save_interval: 5` steps
- **Backend Options**:
  - `safetensors`: Default, portable, CPU-safe
  - `dcp`: PyTorch Distributed Checkpoint (requires world_size>1)
- **Saved Components**:
  - Model state (sharded parameters with FSDP2)
  - Optimizer state (Muon/NorMuon + CautiousAdamW)
  - Scheduler state (WSD or cosine)
  - Training configuration
  - Step count and metadata
- **Async Capability**: Controlled by `hardware.async_checkpointing` (default: true)
  - Overlaps checkpoint I/O with training computation
  - Uses background threads/processes for non-blocking saves

## Throughput Optimizations
### Estimated Performance
From config comments: ≈4.0M tokens/sec on 8×A100 SXM 80GB
- **Per-step tokens**: 1,048,576 (≈1M)
- **Steps per second**: ~4.0 opt-steps/sec
- **150B tokens training**: ~10.4 hours wall-clock

### Bottleneck Analysis
1. **Memory Bandwidth Limited**:
   - Loading weights from HBM to SMs
   - Mitigated by: BF16 precision, activation checkpointing, kernel fusion

2. **Compute Limited**:
   - Matrix multiplications in attention/FFN
   - Mitigated by: TF32, custom kernels, architectural efficiency

3. **Communication Limited**:
   - FSDP all-gather/reduce-scatter operations
   - Mitigated by: `fsdp_forward_prefetch: false`, `fsdp_limit_all_gathers: true`

4. **Kernel Launch Overhead**:
   - Small operations launching many kernels
   - Mitigated by: Kernel fusion, persistent kernels

### Specific Optimizations
#### Kernel Fusion (`training/pretrain.py:fuse_ce_softcap`, `fuse_linear_relu2`)
- **Fused CE + Softcap**:
  - Combines `F.cross_entropy(logits, target)` and `logits = cap * tanh(logits / cap)`
  - Reduces: 2 kernel launches → 1 kernel launch
  - Saves: Memory bandwidth for intermediate logits storage
  - Location: `kernels/ce_softcap.py`

- **Fused Linear + ReLU²**:
  - Replaces `F.relu(x) ** 2` followed by linear projection
  - Location: `kernels/linear_relu2.py`

#### Flash Attention
- **Status**: `use_fa3: false` (disabled by default, BF16 fallback on CPU)
- **Potential**: When enabled, uses FlashAttention-3 for faster attention
- **Location**: Would be integrated in MLA/GDN implementations

#### Tensor Core Utilization
- **BF16**: Native Tensor Core support on Ampere+
- **TF32**: `enable_tf32: true` for faster matmuls with minimal accuracy loss
- **Async Operations**: 
  - `async_checkpointing: true`
  - `async_wandb: true` 
  - `async_mlflow: true`
  - `num_workers: 8` for DataLoader
  - `prefetch_factor: 4` for overlapping I/O and compute

### Scaling Characteristics
#### Weak Scaling (Fixed Problem Size per GPU)
- **Ideal**: Constant time per step as GPUs increase
- **Reality**: 
  - Communication overhead grows with world_size (all-gather bandwidth)
  - Mitigated by: Hierarchical collectives, NVLink, limiting unnecessary gathers

#### Strong Scaling (Fixed Total Problem Size)
- **Ideal**: Time decreases linearly with GPU count
- **Reality**:
  - Sublinear due to communication overhead and load imbalance
  - MoE routing can cause expert imbalance across ranks
  - Mitigated by: Load balancing bias updates, expert capacity factors

### Key Files for Memory/Performance
- Kernel implementations: `kernels/` directory
- Memory-efficient layers: 
  - `models/mla.py` (low-rank KV compression)
  - `models/moe.py` (sparse expert computation)
  - `models/gated_deltanet.py` (efficient SSM)
- Training optimizations: `training/pretrain.py` (checkpointing, fusion, schedulers)
- Memory management: `utils/checkpoint.py`, `utils/distributed.py` (FSDP2 sharding)
- Configuration: `configs/pretrain.yaml` (all knobs)