# Agent Bootstrap: DeepSeek Hybrid Model Repository

## Project Purpose
This repository implements a hybrid language model architecture combining Multi-Head Latent Attention (MLA), Gated Delta Net (GDN/Mamba-2), and Mixture-of-Experts (MoE) for efficient pre-training and inference. Targeting scalable training on 8×RTX 5090 32GB GPUs, it achieves ~7B total / ~2.5B active parameters through MoE sparsity. The design integrates research innovations from DeepSeek-V2/V3 (MLA, MoE, MTP), Qwen3-Next (GDN), and Nemotron-H/Jamba (hybrid schedules) to balance quality, efficiency, and scalability.

## Architecture Summary
- **Hybrid Structure**: Alternating MLA and GDN layers per configurable schedule (default "5:1": 5 MLA + 1 GDN repeated for 30 layers)
  - **MLA Layers**: Multi-Head Latent Attention with low-rank KV compression (q_lora_rank=512, kv_lora_rank=256), GQA (n_heads=32, n_kv_groups=8), decoupled RoPE (qk_nope_head_dim=128, qk_rope_head_dim=64), optional sliding window (2048 tokens with 5:1 local-global interleaving), QK normalization
  - **GDN Layers**: Gated Delta Net (Qwen3-Next style SSM, ssm_type="gdn") or legacy Mamba-2, featuring state dimension (gdn_d_state=128), convolution width (gdn_d_conv=4), head dimension (gdn_headdim=64), and dense SwiGLU FFN
- **MoE Integration**: DeepSeekMoE on MLA layers with 64 routed experts, 6 activated per token, 4 shared experts, group-limited routing (n_expert_groups=8, n_limited_groups=3, group_topk=2), bias updates (speed=1e-3, every=10 steps), expert capacity factor=1.5, SwiGLU activation
- **Enhancements**: 
  - Multi-Token Prediction (MTP) depth=3, loss weight=0.30
  - μP (μ-transfer) re-initialization enabled
  - Tied input/output embeddings (saves ~2×vocab_size×dim parameters)
  - Bias-free linear layers (except MoE gate)
  - Logit softcap (value=15.0) to prevent extreme logits
- **Key Components**: Transformer backbone, ParallelEmbedding (vocab sharding), TransformerBlock (pre-norm residuals), MoE/GDN FFNs, RMSNorm, LM head

## Critical Files
| File | Purpose | Key Sections to Review |
|------|---------|------------------------|
| `models/transformer.py` | Main model backbone | Transformer class, TransformerBlock, ParallelEmbedding, parse_schedule, count_parameters |
| `models/mla.py` | Multi-Head Latent Attention | MultiHeadLatentAttention forward pass, low-rank projections, RoPE application |
| `models/moe.py` | Mixture-of-Experts FFN | DeepSeekMoE forward, group-limited routing, bias updates, expert computation |
| `models/gated_deltanet.py` | GDN SSM implementation | GatedDeltaNet forward, SSM computations |
| `training/pretrain.py` | FSDP2 training loop | Pretrainer class, train_step, _build_optimizers, _maybe_eval, checkpointing |
| `training/normuon.py` | NorMuon optimizer | NorMuon class, Newton-Schulz orthogonalization |
| `utils/distributed.py` | Distributed training setup | setup_distributed, wrap_fsdp2 (FSDP2), collectives (all_reduce_mean, all_to_all), configure_reshard |
| `data/async_loader.py` | Async sharded data loader | AsyncShardLoader for curriculum learning and efficient I/O |
| `data/curriculum.py` | Two-stage curriculum learning | Curriculum class, stage switching, shard management |
| `eval/eval_core.py` | Perplexity evaluation | run_perplexity, make_synthetic_loader (deterministic evaluation) |
| `configs/pretrain.yaml` | Primary training configuration | All hyperparameters for 8×RTX 5090 runs |
| `configs/smoke_pretrain.yaml` | Minimal config for testing | <60s single-GPU smoke tests for validation |

## Training Workflow
1. **Initialization**:
   - Load YAML config → ConfigBundle dataclasses
   - Initialize distributed training (NCCL backend, world_size from torchrun env)
   - Build Transformer model with optional MTP wrapper (shares embed/head)
   - Apply FSDP2 wrapping (per-TransformerBlock, FULL_SHARD strategy, BF16 params/FP32 reduce dtype, backward prefetch on, forward prefetch off, limit_all_gathers=true)
   - Construct dual optimizers: 
     - Muon/NorMuon for matrix parameters (ndim≥2, excluding embed/head) 
     - CautiousAdamW for remaining parameters (embeddings, norms, biases, head)
   - Configure WSD scheduler (0.01 warmup fraction, 0.84 stable fraction, linear decay)
   - Set up gradient scaler (for FP8 if enabled), curriculum learning (if switch_step>0), logging infrastructure (W&B/MLflow/CSV/console), checkpoint manager

2. **Data Pipeline**:
   - Primary: AsyncShardLoader reads from shard manifest (supports curriculum learning)
   - Fallback: PretrainDataset creates packed windows from flat token file (synthetic data generated if missing)
   - Each batch yields (input_tokens, target_tokens) where targets = input rolled left by 1 (next-token prediction)
   - DistributedSampler shards data across ranks when world_size>1
   - Curriculum learning advances stages at switch_step, updating active shards in loader

3. **Per-Iteration Training Step**:
   - Determine if optimizer step needed (every gradient_accumulation_steps micro-batches)
   - Automatic Mixed Precision (AMP) context (BF16 compute)
   - Forward Pass:
     - Embedding lookup: ParallelEmbedding (vocab-sharded, all-reduce if world_size>1)
     - Process through 30 TransformerBlocks (gradient checkpointing if use_checkpoint=true)
       - Each block: Pre-norm RMSNorm → Attention/SSM (MLA or GDN per schedule) → Residual → Pre-norm RMSNorm → FFN (MoE for MLA layers, dense for GDN) → Residual
     - Final RMSNorm → LM Head (weight-tied with embedding if tie_embeddings=true)
   - Loss Computation:
     - Main loss: Cross-entropy between logits and targets (ignore_index=-100)
     - MTP loss: Computed by MTP module if mtp_depth>0 (weighted sum of token t+1, t+2, t+3 predictions)
     - MoE balance loss: Sum of expert auxiliary losses from all MoE layers
     - Total loss = (main_loss + mtp_loss + balance_loss_alpha × balance_loss) / gradient_accumulation_steps
   - Backward Pass:
     - Scale loss, backward() to compute gradients
     - If optimizer step:
       - Unscale gradients for clipping
       - Clip global gradient norm (max_grad_norm=1.0)
       - Step optimizers (Muon/NorMuon then CautiousAdamW)
       - Step learning rate scheduler
       - Zero optimizer gradients (set_to_none=True)
       - Refresh MoE weight stacks after optimizer step
   - Return detached loss metrics for logging

4. **Periodic Operations** (rank 0 only):
   - Logging: Every log_interval steps (default 50) - loss, learning rates, MoE router stats (every 200 steps)
   - Evaluation: Every eval_interval steps (default 1000) - perplexity on synthetic/real data, optional lm-eval-harness tasks
   - Checkpointing: Every save_interval steps (default 500) - model, optimizer, scheduler states + metadata
   - Curriculum Advancement: Check every step for stage switch

## Risks & Mitigations
- **Numerical Instability (NaN/Inf)**:
  - *Mitigations*: logit_softcap=15.0 bounds logits, qk_norm=true stabilizes attention, grad_clip=1.0 prevents exploding gradients, FP32 reduction in softmax/norms
  - *Monitoring*: Check for NaN in loss, gradient norms, expert routing weights
  
- **Out-of-Memory (OOM)**:
  - *Mitigations*: use_checkpoint=true (activation recomputation saves ~30-40% memory), expert_capacity_factor=1.5 prevents expert overflow, activation checkpointing per block
  - *Monitoring*: Track GPU memory usage, activation sizes via profiling
  
- **Distributed Training Hangs**:
  - *Mitigations*: fsdp_limit_all_gathers=true prevents NCCL queue overflow, fsdp_backward_prefetch=true improves consistency, ensure all ranks enter collectives consistently
  - *Mitigation*: Verify world_size matches GPU count, use torchrun with correct env vars (LOCAL_RANK, WORLD_SIZE)
  
- **Expert Load Imbalance**:
  - *Mitigations*: bias_update_speed=1e-3 and bias_update_every=10 adjust expert biases, expert_capacity_factor=1.5 provides headroom
  - *Monitoring*: Logged MoE routing statistics every 200 steps show expert utilization and load balance loss
  
- **Configuration Mismatches**:
  - *Mitigations*: Validate vocab_size matches between model and data config, check sequence length consistency, ensure MoE parameters make sense (n_activated_experts ≤ n_routed_experts)
  - *Validation*: Add startup assertions for critical relationships
  
- **Hardware/Software Compatibility**:
  - *Mitigations*: FP8 matmuls require Blackwell sm_120 and torch≥2.5 (guarded by try/except), verify enable_fp8_mla=false unless confirmed supported
  - *Verification*: Check hardware profile in logs, test with smoke config first
  
- **Curriculum Learning Issues**:
  - *Mitigations*: Ensure shard manifest aligns with actual data files, monitor for data duplication/loss during stage switches
  - *Validation*: Inspect curriculum logs for stage transition events

## Current Status
✅ **Fully Implemented & Tested**:
- Hybrid MLA/GDN architecture with configurable schedules (default 5:1)
- DeepSeekMoE with group-limited routing, bias updates, shared experts
- GDN as default SSM (Qwen3-Next style) with Mamba-2 legacy option (ssm_type)
- Multi-Token Prediction depth=3 with configurable loss weighting
- μP (μ-transfer) re-initialization for stable hyperparameter transfer
- Dual optimizer strategy: NorMuon (matrix params) + CautiousAdamW (rest)
- WSD learning rate scheduler (1% warmup, 84% stable, linear decay)
- Gradient checkpointing per TransformerBlock (memory efficient)
- FSDP2 FULL_SHARD strategy with per-block wrapping
- Tied input/output embeddings, bias-free linear layers (ex: MoE gate)
- Logit softcap (value=15.0), QK normalization in MLA
- Sliding window attention (2048 window) with 5:1 local-global interleaving
- FP8 matmul support (opt-in, requires Blackwell sm_120)
- Two-stage curriculum learning framework
- Comprehensive logging: W&B, MLflow, CSV, console (rank 0 only)
- Robust checkpointing: safetensors (default) and DCP (FSDP2) backends
- Extensive test suite covering data loaders, kernels, models, optimizers, schedulers, distributed utils, checkpointing, curriculum, logging, evaluation, and pipeline integration

⚠️ **Known Limitations / Missing Features**:
- Explicit expert parallelism (relying on FSDP sharding for expert distribution)
- Advanced curriculum (>2 stages) or metric-based switching
- Finer-grained checkpointing options (within TransformerBlock)
- Dynamic architecture adjustment (pruning, width/scaling during training)
- Extended evaluation suite (RULER for long-context, HumanEval, MMLU, GSM8K)
- Tensor/Pipeline parallelism options (FSDP2-only currently)
- Full FP8 training support (beyond matmuls)
- Mixture of Depths (MoD) or adaptive computation time
- Advanced quantization (INT8 quantization-aware training)

## Recommended Next Steps (Immediate Priorities)
1. **Validate at Scale** (0-1 month):
   - Run smoke test: `python -m training.pretrain --config configs/smoke_pretrain.yaml` (<60s validation)
   - Execute small-scale multi-GPU run to verify FSDP2, optimizers, logging
   - Check for NaN/Inf, proper loss decrease, correct optimizer stepping

2. **Stabilize & Observe** (1-2 months):
   - Increase monitoring: Log gradient norms, activation statistics, expert utilization histograms
   - Profile performance: Identify bottlenecks in forward/backward pass, kernel launch overhead
   - Test configuration variations: Different layer schedules (3:1, 6:1, 8:1), MoE parameters, learning rates

3. **Enhance Observability** (2-3 months):
   - Add gradient and activation monitoring to training loop
   - Implement memory profiling utilities to track VRAM breakdown
   - Enhance MoE logging with expert specialization metrics over time
   - Add deadlock detection for distributed settings

4. **Performance Optimization** (3-4 months):
   - Optimize kernels: Fuse additional operations, investigate persistent kernels
   - Improve FSDP2 overlap: Tune prefetching, investigate communication-computation overlap
   - Evaluate alternative optimizers/schedulers via ablation studies
   - Test sequence length scheduling for longer context training

5. **Prepare for Scaling** (4-6 months):
   - Implement explicit expert parallelism for larger expert counts (>128)
   - Extend context length capabilities (RoPE scaling, ALiBi integration)
   - Add support for Tensor Parallelism alongside FSDP2
   - Validate curriculum learning with more than two stages
   - Prepare evaluation suite expansion (standard NLP benchmarks)

## Quick Reference for Agent Work
- **First Files to Read**: `models/transformer.py` (backbone), `training/pretrain.py` (training loop), `configs/pretrain.yaml` (full configuration)
- **Key Interfaces to Preserve**:
  - Attention/SSM: `forward(x) -> [B, S, D]` 
  - FFN: `forward(x) -> [B, S, D]`
  - TransformerBlock: `forward(x) -> [B, S, D]`
  - Transformer: `forward(tokens) -> [B, S, V]`
- **Critical Configuration Sections**: training (optimization, schedules), model (architecture specs), data (loading, curriculum)
- **Common Debugging Steps**:
  1. Check for NaN/Inf in loss → verify logit_softcap, qk_norm, grad_clip
  2. Monitor OOM → enable use_checkpoint, reduce batch/seq length
  3. Watch for expert imbalance → review logged routing stats, adjust bias updates
  4. Diagnose distributed hangs → check fsdp_limit_all_gathers, world_size consistency
  5. Validate configuration mismatches → confirm vocab sizes, sequence lengths, MoE params

This bootstrap provides essential context for productive agent work. For comprehensive details, refer to the complete documentation files in `docs/project_context/`.