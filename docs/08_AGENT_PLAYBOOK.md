# Agent Playbook

This is the most important file for future AI coding agents. It provides guidance on how to approach tasks in this repository, critical components to understand, common failure modes, and safe practices.

## How Future Agents Should Approach Tasks

### 1. First Steps
When approaching any task in this repository:
1. **Check the configuration**: Understand the active config in `configs/pretrain.yaml` or smoke config
2. **Identify the component**: Determine which subsystem your task affects (model, training, data, kernels, etc.)
3. **Read relevant interface files**: Look at the main entry points and public APIs
4. **Check for existing tests**: Though limited, see if there are test patterns to follow
5. **Understand the distributed context**: Most code assumes FSDP2 setup with world_size > 1

### 2. Files to Inspect First
For different types of tasks:

#### Model Architecture Changes
1. `models/transformer.py` - Main model backbone
2. `models/mla.py` - Multi-Head Latent Attention implementation
3. `models/moe.py` - DeepSeekMoE implementation  
4. `models/gated_deltanet.py` - GDN/Mamba-2 SSM layers
5. `models/mtp.py` - Multi-Token Prediction (if applicable)
6. `configs/pretrain.yaml` - To understand current architecture settings

#### Training Pipeline Changes
1. `training/pretrain.py` - Main training loop
2. `training/normuon.py` - NorMuon optimizer implementation
3. `training/schedules.py` - Batch/sequence length scheduling
4. `training/wsd.py` - WSD scheduler
5. `utils/distributed.py` - FSDP2 setup and wrapping
6. `utils/checkpoint.py` - Checkpointing logic

#### Data Pipeline Changes
1. `data/async_loader.py` - Primary data loading mechanism
2. `data/curriculum.py` - Curriculum learning implementation
3. `data/prepare_data.py` - Data preparation (referenced but not shown)
4. `training/pretrain.py:PretrainDataset` - Fallback dataset

#### Kernel Changes
1. `kernels/` directory - All custom CUDA kernels
2. `training/pretrain.py` - Where kernels are used (fuse_* flags)
3. `models/` files - Where kernel functions would be called

#### Evaluation Changes
1. `eval/eval_core.py` - Basic perplexity evaluation
2. `eval/run_lm_eval.py` - LM-eval harness integration
3. `training/pretrain.py:_maybe_eval()` - Evaluation calling logic

### 3. Critical Components to Understand

#### The Transformer Block (`models/transformer.py:TransformerBlock`)
- **Core abstraction**: Alternates between MLA and GDN layers
- **Key features**:
  - Per-block FSDP2 wrapping granularity
  - Optional gradient checkpointing (`use_checkpoint`)
  - MoLE every Nth FFN slot capability
  - Dual FFN paths: MoE (for MLA) vs dense (for GDN)
  - Pre-normalization architecture
  - Residual connections around attention/SSM and FFN

#### The MLA Implementation (`models/mla.py`)
Understanding this is crucial as it's a key innovation:
- **Low-rank KV compression**: QKV projections via LoRA
- **Decoupled RoPE**: Separate treatment of content and positional dimensions
- **Grouped Query Attention**: Multiple queries share KV heads
- **Optional sliding window**: For local-global attention patterns
- **QK normalization**: Stabilizes training

#### The MoE Implementation (`models/moe.py`)
Critical for efficiency:
- **Group-limited routing**: Experts organized in groups for efficiency
- **Bias updates**: Aux-loss-free load balancing mechanism
- **Shared experts**: Always-active expert subset
- **Capacity factors**: Prevent expert overflow
- **Routing statistics**: Important for monitoring and debugging

#### The FSDP2 Implementation (`utils/distributed.py:wrap_fsdp2`)
Essential for distributed training:
- **Per-block wrapping**: TransformerBlock granularity
- **Mixed precision**: BF16 params, FP32 reduce
- **Prefetching strategy**: Backward prefetch on, forward off
- **Resharding configuration**: `configure_reshard()` for backward pass optimization
- **Collective operations**: Custom wrappers for all-reduce, all-to-all, etc.

#### The Dual Optimizer System (`training/pretrain.py:_build_optimizers`)
Unique optimization approach:
- **Muon/NorMuon**: For matrix parameters (weights with ndim ≥ 2)
- **CautiousAdamW**: For everything else (embeddings, norms, biases, head)
- **Parameter separation logic**: Based on dimensionality and name matching
- **Logging**: Parameter counts reported per optimizer type

#### The WSD Scheduler (`training/wsd.py` and `training/pretrain.py`)
Learning rate schedule:
- **Three phases**: Linear warmup → stable LR → decay
- **Configurable fractions**: Warmup, stable, decay portions of training
- **Decay types**: Linear or cosine to minimum LR ratio
- **Purpose**: Stabilizes training at large scale

### 4. Common Failure Modes

#### Out-of-Memory (OOM) Errors
**Likely locations**:
- During model initialization (too large for GPUs)
- Forward pass with large batch/sequence length
- Backward pass with gradient checkpointing disabled
- Expert capacity exceeded in MoE layers

**Diagnosis**:
- Check GPU memory usage before/after operations
- Look for patterns in batch_size × seq_len × world_size
- Verify `use_checkpoint: true` is enabled
- Check `expert_capacity_factor` and logged expert loads

**Solutions**:
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Increase expert capacity factor
- Use sequence length scheduling to start small
- Monitor activation memory with profiling tools

#### Numerical Instabilities (NaN/Inf)
**Likely locations**:
- Attention softmax with large logits
- MoE gate logits becoming extreme
- Muon optimizer Newton-Schulz divergence
- FP8matmul underflow/overflow (when enabled)

**Diagnosis**:
- Check training logs for sudden loss jumps to NaN
- Monitor gradient norms if logging enabled
- Check for Inf values in expert routing weights
- Verify logit softcap is enabled (`logit_softcap: 15.0`)

**Solutions**:
- Enable/logit softcap (`logit_softcap: 15.0`)
- Check QK normalization (`qk_norm: true`)
- Reduce learning rate if instability early in training
- Disable FP8 matmuls if using (`use_fp8_mla: false`)
- Check gradient clipping (`grad_clip: 1.0`)

#### Distributed Training Hangs
**Likely locations**:
- FSDP parameter all-gather operations
- Expert all-to-all in MoE routing (if implemented)
- Collective operations in logging/checkpointing
- Curriculum learning shard switching

**Diagnosis**:
- Check if all ranks are progressing or some stuck
- Look at NCCL errors in logs
- Verify world_size matches actual GPU count
- Check for deadlocks in custom all-to-all implementations
- Ensure consistent code paths across ranks

**Solutions**:
- Verify `torchrun` launch with correct environment
- Check `fsdp_limit_all_gathers: true` to prevent queue overflow
- Ensure `shard_keep_last` is set appropriately (default 1)
- Verify all ranks enter collective calls consistently
- Check for rank-0 only operations that should be collective

#### Poor Expert Load Balancing
**Likely locations**:
- MoE gate bias updates not working
- Expert capacity exceeded causing token dropping
- Routing collapse to few experts

**Diagnosis**:
- Check logged MoE routing statistics (every 200 steps)
- Look for experts with 0% utilization or >capacity%
- Verify bias update frequency (`bias_update_every: 10`)
- Check expert capacity factor (`expert_capacity_factor: 1.5`)
- Monitor loss balance loss component

**Solutions**:
- Increase expert capacity factor
- Adjust bias update speed (`bias_update_speed: 1e-3`)
- Increase number of experts or activated experts per token
- Check routing logic for bugs
- Ensure shared experts are functioning

#### Configuration Mismatches
**Likely locations**:
- Vocab size mismatch between model and data
- Sequence length mismatch in different components
- FP8 enabled without Blackwell hardware
- MoE parameters inconsistent (experts < activated)

**Diagnosis**:
- Check initialization error messages
- Verify `model.vocab_size` matches `data.vocab_size`
- Check sequence lengths in config vs data loader
- Validate FP8 hardware support before enabling
- Check MoE parameter relationships

**Solutions**:
- Ensure consistent configuration across sections
- Validate config during startup (add assertions)
- Use smoke config to test changes before full runs
- Document assumed relationships in comments

### 5. Safe Refactoring Guidelines

#### Making Changes to Model Architecture
1. **Maintain interface contracts**:
   - TransformerBlock `forward(x) -> x` must preserve shape
   - Attention/SSM modules must accept `(config, layer_idx, world_size, rank)`
   - FFN modules must accept input hidden states and return same shape
   
2. **Preserve initialization patterns**:
   - Use `_init_weights` function or equivalent
   - Consider μP re-initialization if applicable
   - Maintain tied embedding behavior if changing embedding/head

3. **Keep FSDP2 compatibility**:
   - Don't change module hierarchy that breaks per-block wrapping
   - Avoid operations that don't work with activation checkpointing
   - Test with both world_size=1 and world_size>1

4. **Maintain configuration consistency**:
   - Add new config options to `ConfigBundle` dataclasses
   - Provide sensible defaults in YAML configs
   - Update both `pretrain.yaml` and `smoke_pretrain.yaml` when relevant

#### Making Changes to Training Pipeline
1. **Preserve optimizer contracts**:
   - Muon/NorMuon should only get matrix parameters (ndim ≥ 2, not embed/head)
   - CautiousAdamW should get the rest
   - Maintain learning rate scheduling interface

2. **Keep evaluation logic intact**:
   - Evaluation should be rank-0 only
   - Maintain synthetic data fallback path
   - Preserve W&B/MLflow/CSV logging structure

3. **Maintain checkpoint compatibility**:
   - Don't change saved metadata structure without versioning
   - Consider backward compatibility for resuming
   - Test loading old checkpoints if changing format

4. **Preserve gradient accumulation logic**:
   - Effective batch size calculations should remain consistent
   - Optimizer stepping should happen at correct micro-step boundaries
   - Gradient scaling should work with AMP/FP8

#### Making Changes to Kernels
1. **Maintain numerical equivalence**:
   - New kernels should produce same results (within FP tolerance)
   - Test with deterministic inputs
   - Verify gradients match if applicable

2. **Preserve interface contracts**:
   - Match input/output tensor shapes and dtypes
   - Keep same function signatures
   - Maintain device placement expectations

3. **Consider performance characteristics**:
   - Profile new kernels against old ones
   - Check occupancy, memory bandwidth utilization
   - Ensure block/grid sizes are appropriate

4. **Maintain build system compatibility**:
   - Ensure kernels can be compiled with nvcc
   - Check dependencies on CUDA toolkit version
   - Verify inclusion in build process

#### Making Changes to Data Pipeline
1. **Preserve data contracts**:
   - Dataset should yield `(tokens, targets)` with `targets = tokens[:, 1:]`
   - Handle `ignore_index=-100` appropriately in loss functions
   - Maintain sequence length consistency

2. **Keep sharding logic sound**:
   - DistributedSampler should work when world_size > 1
   - AsyncShardLoader should maintain correct sharding
   - Curriculum learning should not cause data duplication/loss

3. **Maintain serialization compatibility**:
   - Don't change saved data format without migration path
   - Consider versioning for data files
   - Test loading existing data with new code

### 6. Training Debugging Workflow

#### When Loss Looks Wrong
1. **Check if NaN/Inf**:
   - If yes, look for numerical instability sources
   - Check logit softcap, QK norm, gradient clipping

2. **If loss not decreasing**:
   - Verify learning rate schedule is correct
   - Check optimizer parameter separation
   - Look at gradient norms if available
   - Check for bugs in loss computation

3. **If loss decreasing too slowly/fast**:
   - Check effective batch size calculation
   - Verify data loading is working
   - Check curriculum learning if enabled
   - Look at data quality/synthetic fallback

#### When Seeing Poor Performance
1. **Check utilization**:
   - GPU utilization % (should be high)
   - Memory utilization (look for unexpected spikes)
   - Kernel timings if profiling available

2. **Look for bottlenecks**:
   - FSDP all-gather time (check NCCL logs)
   - Kernel launch overhead (too many small kernels)
   - Memory bandwidth limitations
   - Load imbalance (check expert stats)

3. **Check configuration**:
   - Batch size and sequence length settings
   - Gradient checkpointing enabled
   - Kernel fusion flags
   - Data loading efficiency

#### When Encountering OOM
1. **Check memory breakdown**:
   - Static state (parameters, optimizer states)
   - Activation memory (affected by checkpointing)
   - KV cache size (affected by sequence length)
   - Fragmentation and overhead

2. **Apply remedies in order**:
   - Enable gradient checkpointing (`use_checkpoint: true`)
   - Reduce batch size
   - Reduce sequence length
   - Increase expert capacity factor (if MoE OOM)
   - Check for memory leaks in custom code

#### When Seeing Expert Issues
1. **Check routing statistics**:
   - Look at logged expert utilization (every 200 steps)
   - Check for experts with 0% or near-capacity%
   - Look at load balance loss component

2. **Apply remedies**:
   - Increase expert capacity factor
   - Adjust bias update speed/frequency
   - Check routing algorithm implementation
   - Verify shared experts are working
   - Consider increasing n_routed_experts or n_activated_experts

### 7. Performance Debugging Workflow

#### When Throughput Seems Low
1. **Measure baseline**:
   - Tokens per second (should be ~1.5M for 8×RTX 5090)
   - Steps per second
   - Compare to expectations in config comments

2. **Profile components**:
   - Forward pass time
   - Backward pass time
   - Optimizer step time
   - Data loading time
   - Logging/checkpointing time

3. **Check for common issues**:
   - FSDP all-gather inefficiency (check `fsdp_limit_all_gathers`)
   - Excessive kernel launches (check fusion flags)
   - Memory bandwidth saturation
   - Load imbalance (MoE routing or data sharding)

#### When Memory Usage Seems High
1. **Break down usage**:
   - Model parameters (should be sharded)
   - Optimizer states (should be sharded)
   - Gradients (should be sharded)
   - Activations (should be managed by checkpointing)
   - KV cache (check sequence length)
   - PyTorch caching allocator overhead

2. **Check configuration**:
   - Gradient checkpointing enabled
   - Sequence length not excessively large
   - Batch size appropriate for memory
   - Expert capacity factor reasonable

#### When Seeing Slow Convergence
1. **Check optimization**:
   - Learning rate schedule correct
   - Optimizer separation working
   - Gradient clipping not too aggressive
   - Weight decay applied correctly

2. **Check data quality**:
   - Not stuck in synthetic data fallback
   - Curriculum learning working if enabled
   - Data mixing ratios correct
   - Token distribution reasonable

3. **Check model capacity**:
   - Not under/over-sized for task
   - Architecture appropriate for data
   - Pretraining duration sufficient

### 8. Evaluation Workflow

#### Running Evaluation
1. **Synthetic evaluation (fast)**:
   - Set `eval_enabled: true`
   - Set `eval_synthetic: true` (default)
   - Set `eval_interval: desired frequency`
   - Uses deterministic random data loader

2. **Real evaluation (slower)**:
   - Requires `lm_eval` package installed
   - Set `eval_synthetic: false`
   - Requires validation data at `data/validation_data.bin`
   - Will run specified tasks from `eval.tasks`

#### Interpreting Results
1. **Perplexity metrics**:
   - Lower is better (exponential of average cross-entropy)
   - Compare to baseline or previous checkpoints
   - Watch for sudden jumps indicating instability

2. **Task-specific metrics**:
   - Hellaswag, ARC-c, etc. (higher is better)
   - Look for improvement over training
   - Check for task correlation (some should improve together)

3. **Evaluation frequency tradeoffs**:
   - More frequent: Better monitoring, less training time
   - Less frequent: More training time, coarser monitoring
   - Default `eval_interval: 1000` steps reasonable for 50k step runs

#### Troubleshooting Evaluation
1. **If evaluation fails**:
   - Check if rank-0 only condition met (`is_main_process()`)
   - Verify data loader is working
   - Check model is in eval mode during evaluation
   - Look for OOM during evaluation (may need smaller batch)

2. **If results seem wrong**:
   - Check tokenization matches between training and eval
   - Verify `ignore_index=-100` handling
   - Check for data contamination issues
   - Verify evaluation logic matches training loss computation

### 9. Quick Reference for Common Tasks

#### Adding a New Model Component
1. Create file in `models/` directory (e.g., `models/new_component.py`)
2. Implement required interface (forward preserving shapes)
3. Import in `models/transformer.py` or relevant file
4. Add configuration options to `ConfigBundle` if needed
5. Update `configs/pretrain.yaml` with defaults
6. Test with both world_size=1 and world_size>1
7. Verify activation checkpointing compatibility
8. Check parameter counting includes new component

#### Modifying Training Loop Logic
1. Understand current flow in `training/pretrain.py:train_step()` and `train()`
2. Identify where change belongs (forward, loss, backward, step)
3. Maintain gradient accumulation boundary logic
4. Preserve optimizer stepping frequency
5. Keep evaluation and checkpointing calls intact
6. Test with small smoke run first
7. Verify logging still works correctly

#### Changing Configuration Structure
1. Add field to appropriate `ConfigBundle` dataclass in `training/pretrain.py`
2. Provide default value in dataclass field definition
3. Add to `build_config_from_yaml()` mapping if from YAML
4. Update `configs/pretrain.yaml` with example/default
5. Update `configs/smoke_pretrain.yaml` if relevant
6. Check all places that use the configuration
7. Ensure backward compatibility if changing existing field

#### Adding a New Kernel
1. Create file in `kernels/` directory (e.g., `kernels/new_kernel.py`)
2. Implement CUDA kernel with proper error checking
3. Create Python wrapper with torch.autograd.Function if needed
4. Ensure proper input/output tensor handling
5. Add build instructions if needed (may require setup.py changes)
6. Import and use in relevant model/training code
7. Add configuration flag to enable/disable
8. Test numerical equivalence to existing implementation
9. Profile performance improvement

#### Debugging Distributed Issues
1. Start with world_size=1 to isolate local issues
2. Gradually increase world_size to find scaling problems
3. Check NCCL backend availability and version
4. Verify environment variables (LOCAL_RANK, WORLD_SIZE, etc.)
5. Use torch.distributed debugging tools if available
6. Check for rank-0 only assumptions in code
7. Ensure all ranks enter collective operations consistently
8. Look for deadlocks in custom all-to-all implementations

## Conclusion
This repository implements a sophisticated hybrid language model architecture with many interconnected components. Successful modifications require understanding:
1. The modular design (separable model, training, data, kernel layers)
2. The distributed training assumptions (FSDP2-centric)
3. The configuration-driven nature of behavior
4. The importance of maintaining interface contracts
5. The unique optimization strategies (dual optimizer, WSD scheduler, etc.)

By following this playbook, future agents can make safe, effective changes while avoiding common pitfalls in this complex system.