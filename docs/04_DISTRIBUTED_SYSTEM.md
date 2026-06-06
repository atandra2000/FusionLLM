# Distributed System

## FSDP Wrapping
The project uses FSDP2 (``torch.distributed.fsdp.fully_shard``) exclusively, with no support for DDP. This is a deliberate design choice targeting the canonical 8×A100 SXM 80GB RunPod node.

### Wrapping Policy (utils/distributed.py:wrap_fsdp2)
- **Granularity**: Per-TransformerBlock auto-wrap using `ModuleWrapPolicy`
- **Rationale**: 
  - Good default for hybrid MLA + Mamba-2 + MoE backbones
  - Expert linears inside DeepSeekMoE are sharded as part of surrounding block
  - No per-expert wrapping attempted (experts are small enough for uniform block-level sharding)
- **Fallback**: If no TransformerBlock found, wraps entire model (single FSDP unit)

### FSDP2 Configuration
Parameters come from training config (`configs/pretrain.yaml:training`):
- `fsdp_shard_strategy: FULL_SHARD` (shards params, grads, optimizer states)
- `fsdp_forward_prefetch: false` (saves H2D bandwidth on hot path)
- `fsdp_backward_prefetch: true` (PyTorch 2.4+ default for performance)
- `fsdp_limit_all_gathers: true` (prevents NCCL queue depth buildup)
- `fsdp_param_dtype: bf16` (parameter storage dtype)
- `fsdp_reduce_dtype: fp32` (gradient reduction dtype for numerical stability)
- `shard_keep_last: 1` (Phase 3.2: keeps last N FSDP units resident)

### Mixed Precision Policy
Created from `fsdp_param_dtype` and `fsdp_reduce_dtype`:
- Stores parameters in BF16 to reduce memory footprint
- Reduces gradients in FP32 for numerical stability
- Uses `torch.distributed.fsdp.MixedPrecisionPolicy`

## Sharding Strategy
### FULL_SHARD Behavior
- **Parameters**: Sharded across all ranks (world_size-way)
- **Gradients**: Sharded during backward pass
- **Optimizer States**: Sharded (optimizer-specific moments per shard)
- **Forward Pass**: Requires all-gather of sharded parameters before layer computation
- **Backward Pass**: Reduce-scatter gradients, then all-gather for next layer

### Communication Patterns
1. **Forward Pass** (per TransformerBlock):
   - All-gather: Collect full parameter shards from all ranks
   - Compute: Local matrix operations with gathered parameters
   - No communication: Pointwise operations (activations, residuals)

2. **Backward Pass** (per TransformerBlock):
   - Reduce-scatter: Distribute gradient contributions across ranks
   - All-gather: Gather full input activations for gradient computation
   - Compute: Local gradient w.r.t. parameters
   - All-reduce: Synchronize gradient reductions (handled by FSDP internals)

3. **Optimizer Step**:
   - Each rank updates only its parameter shards
   - No cross-rank communication needed for optimizer step

### Expert Parallelism Considerations
While not explicitly implemented as separate expert parallelism, the MoE layers benefit from FSDP sharding:
- **DeepSeekMoE Experts**: Each expert's weights are sharded alongside other block parameters
- **Routing**: Tokens routed to experts based on gate scores (computed locally after parameter all-gather)
- **Expert Computation**: Each rank computes эксперты for its assigned tokens using local expert weights
- **Combine**: Results combined via all-reduce-like operations in the MoE module

### Expected Cluster Topology
Designed for single-node multi-GPU:
- **Canonical Target**: 1× / 8× NVIDIA A100 SXM 80GB on RunPod
- **Interconnect**: NVLink (checked via `hardware.enable_nvlink_check: true`)
- **Communication Backend**: NCCL (hardcoded in `setup_distributed`)
- **Process Layout**: 
  - 1 process per GPU (world_size = n_gpus)
  - Local ranks map 1:1 to physical GPUs
  - No multi-node support explicitly documented but should work via standard `torchrun` env-vars

### Scaling Assumptions
From `configs/pretrain.yaml` comments:
- **Per-step tokens per rank**: 
  `micro_batch_size × gradient_accumulation_steps × max_seq_len`
  `= 2 × 16 × 4096 = 131,072`
- **Total per-step tokens (world_size=8)**: 
  `131,072 × 8 = 1,048,576 ≈ 1M tokens/opt-step`
- **Throughput**: ≈4.0M tokens/sec on 8×A100 SXM 80GB
- **150B tokens training time**: ≈10.4 hours wall-clock on single 8×A100 SXM 80GB node

### Performance Bottlenecks
1. **Parameter All-Gather** (Forward Pass):
   - Scales O(world_size) bandwidth per layer
   - Mitigated by: `fsdp_forward_prefetch: false` (saves H2D bandwidth)
   - Peak during: Attention/SSM computation phases

2. **Gradient Reduce-Scatter** (Backward Pass):
   - Similar all-to-all communication pattern
   - Peak during: Backward pass through large weight matrices

3. **Address Bottlenecks**:
   - **Attention QKV Projections**: Large `[dim, dim]` matrices
   - **FFN Projections**: `[dim, inter_dim]` and `[inter_dim, dim]` matrices
   - **Embedding Lookups**: Reduced by vocab sharding (`ParallelEmbedding`)

4. **Optimizations**:
   - **Selective Prefetching**: Backward prefetch enabled, forward disabled
   - **NCCL Queue Limiting**: `fsdp_limit_all_gathers: true`
   - **Resharding Tuning**: `shard_keep_last: 1` reduces backward all-gather pressure

### Memory Characteristics
As noted in config comments:
- **Static State per GPU**: ~4.5 GB on 8×A100 SXM 80GB with FSDP2 FULL_SHARD
- **Activation Memory**: Managed by gradient checkpointing (`use_checkpoint: true`)
- **KV Cache**: Dynamically allocated; reduced by MLA low-rank compression
- **Headroom**: Activations + KV cache should fit comfortably in remaining VRAM

### Key Files
- Distributed setup: `utils/distributed.py`
- FSDP2 wrapping: `utils/distributed.py:wrap_fsdp2()`
- Re-sharding config: `utils/distributed.py:configure_reshard()`
- Process group init: `utils/distributed.py:setup_distributed()`
- Collectives: `utils/distributed.py:all_reduce_mean()`, etc.
- Integration: `training/pretrain.py` (calls setup, wrap, optimizers)