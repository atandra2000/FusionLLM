# Agent Context: DeepSeek Hybrid Model Repository

## Essential Information for Productive Agent Work (5-Minute Read)

### 🎯 Repository Purpose
Implements a hybrid language model combining Multi-Head Latent Attention (MLA), Gated Delta Net (GDN/Mamba-2), and Mixture-of-Experts (MoE) for efficient pre-training and inference.

### ⚙️ Core Architecture
- **Hybrid Structure**: 5 MLA layers + 1 GDN layer repeating (configurable via `layer_schedule`)
- **MLA**: Low-rank KV compression, GQA, decoupled RoPE, optional sliding window
- **MoE**: DeepSeekMoE with 64 experts, 6 activated, 4 shared, group-limited routing
- **SSM**: GDN (default) or Mamba-2 (legacy) for every 6th layer
- **Additional**: MTP depth=3, μP re-initialization, tied embeddings, bias-free linears
- **Params**: ~7B total / ~2.5B active (due to MoE sparsity)

### 📂 Key Directories
```
models/       # Neural network components (Transformer, MLA, MoE, GDN, etc.)
training/     # Training loop, optimizers, schedulers
data/         # Data loading (AsyncShardLoader), curriculum learning
kernels/      # Custom CUDA kernels (fused ops, FP8 matmuls)
eval/         # Evaluation harness (perplexity, lm-eval)
utils/        # Checkpointing, distributed training (FSDP2), logging
configs/      # YAML configuration files
```

### 🚀 Training Pipeline Essentials
- **Optimizer**: Muon/NorMuon for matrix params (≥2D, not embed/head) + CautiousAdamW for rest
- **Scheduler**: WSD (Warmup-Stable-Decay) by default: 1% warmup, 84% stable, linear decay
- **Precision**: BF16 compute, FP32 gradient reduction (FSDP2)
- **Parallelism**: FSDP2 only (no DDP), per-TransformerBlock wrapping
- **Memory**: Gradient checkpointing enabled, activation recomputation
- **Batch**: Effective batch = micro_batch_size × grad_accum × world_size
- **Logging**: W&B + MLflow + CSV (rank 0 only)
- **Evaluation**: Perplexity + optional lm-eval-harness (Hellraswag, ARC-c, etc.)

### ⚠️ Critical Files to Modify
| Task | Files to Inspect First |
|------|------------------------|
| Model Architecture | `models/transformer.py`, `models/mla.py`, `models/moe.py`, `models/gated_deltanet.py` |
| Training Logic | `training/pretrain.py` (train_step, train methods) |
| Optimizers | `training/pretrain.py:_build_optimizers()`, `training/normuon.py` |
| Data Loading | `data/async_loader.py`, `data/curriculum.py` |
| Kernels | `kernels/ce_softcap.py`, `kernels/linear_relu2.py` |
| Distributed | `utils/distributed.py` (FSDP2 wrapping, collectives) |
| Configuration | `configs/pretrain.yaml`, `training/pretrain.py:ConfigBundle` |
| Evaluation | `eval/eval_core.py`, `eval/run_lm_eval.py` |

### 🔧 Common Failure Modes & Solutions
1. **OOM Errors**: 
   - Enable `use_checkpoint: true` (gradient checkpointing)
   - Reduce batch size or sequence length
   - Increase `expert_capacity_factor` (MoE)
   
2. **Numerical Instability (NaN/Inf)**:
   - Ensure `logit_softcap: 15.0` is enabled
   - Verify `qk_norm: true` in MLA config
   - Check `grad_clip: 1.0` is set
   
3. **Distributed Hangs**:
   - Confirm `fsdp_limit_all_gathers: true`
   - Verify world_size matches GPU count
   - Ensure consistent collective calls across ranks
   
4. **Poor Expert Load Balancing**:
   - Check logged routing stats (every 200 steps)
   - Adjust `bias_update_speed: 1e-3` or `bias_update_every: 10`
   - Increase `expert_capacity_factor: 1.5`

### 🛡️ Safe Refactoring Guidelines
1. **Preserve Interface Contracts**:
   - Attention/SSM: `forward(x) -> [batch, seq_len, dim]`
   - FFN: `forward(x) -> [batch, seq_len, dim]`
   - TransformerBlock: `forward(x) -> [batch, seq_len, dim]`
   - Transformer: `forward(tokens) -> [batch, seq_len, vocab_size]`

2. **Maintain Configuration Consistency**:
   - Add new options to `ConfigBundle` dataclasses
   - Provide defaults in YAML configs
   - Update both `pretrain.yaml` and `smoke_pretrain.yaml`

3. **Keep Distributed Compatibility**:
   - Test with world_size=1 and world_size>1
   - Avoid breaking FSDP2 wrapping assumptions
   - Preserve gradient accumulator boundary logic

4. **Preserve Checkpointing**:
   - Don't change saved metadata structure without versioning
   - Test loading existing checkpoints after changes

### 📊 Key Configuration Parameters (`configs/pretrain.yaml`)
**Training**:
- `micro_batch_size: 2` (per rank)
- `gradient_accumulation_steps: 16`
- `total_steps: 50_000`
- `lr: 3e-4` (AdamW base)
- `muon_lr: 0.02` (Muon/NorMuon)
- `use_checkpoint: true`
- `eval_enabled: false` (set true for monitoring)
- `fsdp_shard_strategy: FULL_SHARD`

**Model**:
- `vocab_size: 152064` (Qwen2.5)
- `max_seq_len: 4096`
- `dim: 2048`
- `n_layers: 30`
- `layer_schedule: "5:1"` (5 MLA + 1 GDN)
- `n_heads: 32`, `n_kv_groups: 8` (GQA)
- `n_routed_experts: 64`, `n_activated_experts: 6`, `n_shared_experts: 4`
- `mtp_depth: 3`, `mtp_loss_weight: 0.30`
- `muP: true`

### 🧪 Quick Validation Workflow
1. **Smoke Test**: Use `configs/smoke_pretrain.yaml` (runs in <60s on 1 GPU)
2. **Single GPU**: `python -m training.pretrain --config configs/smoke_pretrain.yaml`
3. **Multi-GPU**: `torchrun --nproc_per_node=8 training/pretrain.py --config configs/pretrain.yaml`
4. **Check Logs**: Rank 0 outputs loss, learning rates, MoE routing stats
5. **Verify**: No NaN/Inf, decreasing loss, proper optimizer stepping

### 💡 Agent Best Practices
- Start with `08_AGENT_PLAYBOOK.md` for detailed task guidance
- Use configuration flags rather than hardcoding changes
- Preserve existing architecture patterns when adding features
- Test changes with smoke config before full runs
- Monitor expert utilization and loss components
- Keep backward compatibility for checkpoint resuming
- Log numerical stability metrics (grad norms, activation stats)

This condensed guide provides the minimum context needed to make safe, effective changes to this repository. For deeper understanding, refer to the full documentation files in `docs/project_context/`.