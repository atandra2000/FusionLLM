# Research Roadmap

## Current Architecture Status
The repository implements a hybrid language model architecture combining:
- **Multi-Head Latent Attention (MLA)** with low-rank KV compression
- **Gated Delta Net (GDN)** or Mamba-2 SSM layers in configurable schedules
- **DeepSeekMoE** fine-grained mixture-of-experts with group-limited routing
- **Multi-Token Prediction (MTP)** with depth=3
- **μP (μ-transfer)** re-initialization for stable scaling
- **Advanced optimizations**: NorMuon optimizer, WSD scheduler, cautious weight decay
- **Efficient training**: FSDP2 sharding, gradient checkpointing, kernel fusion

### Verified Implementation
From code analysis, the following features are fully implemented and functional:
- Hybrid MLA/GDN architecture with 5:1 schedule (5 MLA + 1 GDN layers)
- DeepSeekMoE with 64 routed experts, 6 activated, 4 shared
- Group-limited routing (8 groups, top-3 groups, 2 experts per group)
- Gated Delta Net (Qwen3-Next style) as default SSM
- Multi-Token Prediction depth=3
- μP re-initialization enabled
- NorMuon + CautiousAdamW dual optimizer strategy
- WSD learning rate scheduler
- Gradient checkpointing per TransformerBlock
- FSDP2 FULL_SHARD strategy
- Bias-free linear layers (except MoE gate)
- Tied input/output embeddings
- Logit softcap (value=15.0)
- QK normalization in MLA
- Sliding window attention with 5:1 local-global interleaving
- Curriculum learning framework
- Comprehensive logging (W&B, MLflow, CSV)
- Checkpointing with safetensors and DCP backends

## Missing Features
Based on the codebase analysis and comments, the following features are planned but not yet implemented:

### 1. Expert Parallelism
**Status**: Not explicitly implemented; relying on FSDP sharding for expert distribution
**Location**: Missing explicit expert parallelism utilities
**Priority**: Medium-High (important for very large expert counts)
**Implementation Path**:
- Add explicit expert parallelism utilities
- Modify MoE layer to support expert parallelism groups
- Update collective operations for expert all-to-all
- Consider hierarchical expert+data parallelism

### 2. Advanced Curriculum Learning
**Status**: Basic two-stage curriculum implemented
**Missing**:
- More than two stages
- Dynamic switching based on metrics
- Curriculum over architectural parameters (not just data)
**Priority**: Medium
**Implementation Path**:
- Extend `data/curriculum.py` to support N stages
- Add metric-based switching criteria
- Consider curriculum over model width/depth

### 3. Activation Recomputation Options
**Status**: Per-TransformerBlock gradient checkpointing
**Missing**:
- Finer-grained checkpointing within blocks
- Selective checkpointing of specific components
- Memory profiling to guide checkpointing decisions
**Priority**: Low-Medium
**Implementation Path**:
- Add options to checkpoint attention vs FFN separately
- Provide memory usage reporting
- Allow selective activation recomputation

### 4. Dynamic Architecture Adjustment
**Status**: Fixed architecture parameters
**Missing**:
- Pruning of heads, experts, or layers during training
- Dynamic width adjustment based on importance
- Neural architecture search integration
**Priority**: Low
**Implementation Path**:
- Importance scoring for heads/experts
- Pruning schedules and fine-tuning
- Architecture search controllers

### 5. Advanced Evaluation Suite
**Status**: Basic perplexity and lm-eval-harness integration
**Missing**:
- RULER for long-context evaluation
- HumanEval style code generation benchmarks
- MMLU, GSM8K, and other standard benchmarks
- Throughput and latency measurements
**Priority**: Medium
**Implementation Path**:
- Extend `eval/run_lm_eval.py` to support more task types
- Add benchmark suites for specific capabilities
- Implement latency/throughput measurement utilities

### 6. Model Parallelism Options
**Status**: FSDP2 only ( tensor parallelism not implemented)
**Missing**:
- Tensor Parallelism (TP) for very large models
- Pipeline Parallelism (PP) options
- Expert Parallelism (EP) as mentioned above
**Priority**: Medium (for scaling beyond current target)
**Implementation Path**:
- Add TP/PP/EP wrapping options
- Modify distributed utilities to support hybrid parallelism
- Update configuration for parallelism specs

### 7. Advanced Quantization Support
**Status**: BF16/FP16
**Missing**:
- Full FP8 training
- INT8 quantization for inference
- Quantization-aware training (QAT)
**Priority**: Low-Medium
**Implementation Path**:
- Add FP8 training support
- Add INT8 quantization utilities
- Implement QAT workflows

### 8. Mixture of Depths (MoD)
**Status**: Fixed number of layers (30)
**Missing**:
- Mixture of Depths layer skipping
- Computation savings from dynamic depth
**Priority**: Low
**Implementation Path**:
- Learnable layer skipping mechanisms
- Routing over depth similar to expert routing
- Adaptive computation time

## Improvement Opportunities
Based on code review, the following areas present opportunities for enhancement:

### 1. Memory Optimization
**Opportunities**:
- More aggressive activation checkpointing
- Improved KV cache compression strategies
- Better memory allocator settings
- Gradient compression for FSDP
**Impact**: Could reduce memory footprint by 20-40%
**Effort**: Low-Medium

### 2. Numerical Stability
**Opportunities**:
- Additional safeguards in Muon optimizer
- Improved handling of extreme values in MoE routing
- More robust FP8 support
- Better gradient clipping strategies
**Impact**: Improves training reliability at scale
**Effort**: Low

### 3. Performance Optimization
**Opportunities**:
- Kernel fusion for more operations
- Persistent kernels to reduce launch overhead
- Better overlap of communication and computation
- Optimized all-to-all patterns for MoE
**Impact**: Could improve throughput by 10-30%
**Effort**: Medium-High

### 4. Observability and Debugging
**Opportunities**:
- Gradient and activation statistics logging
- Expert utilization tracking over time
- Memory usage profiling
- Deadlock detection in distributed settings
**Impact**: Significantly improves debuggability
**Effort**: Low-Medium

### 5. Code Quality and Maintainability
**Opportunities**:
- Consolidate duplicate configuration structures
- Improve error handling consistency
- Add more comprehensive unit tests
- Better documentation of complex algorithms
**Impact**: Reduces technical debt, improves contribution
**Effort**: Medium

## Scaling Roadmap
The architecture is designed to scale efficiently within its target domain:

### Short-Term (0-6 months)
**Target**: Stable 7B parameter model training on 8×A100 SXM 80GB
**Focus**:
- Validate current architecture at scale
- Extend evaluation suite
- Improve observability and debugging tools
- Optimize kernel performance
- Validate μP transfer effectiveness
**Milestones**:
- Stable 150B token training run
- Comprehensive evaluation benchmark suite
- Published technical report on findings

### Medium-Term (6-18 months)
**Target**: Scaling to larger parameter counts (14B-70B) and longer contexts
**Focus**:
- Expert parallelism implementation
- Context length extension (beyond 4096)
- Architecture search for optimal layer schedules
- Mixed precision beyond BF16/FP16
- Multi-node distributed training
**Milestones**:
- 14B parameter model training
- 32K context length support
- Multi-node scaling demonstrations

### Long-Term (18+ months)
**Target**: Frontier model capabilities and efficiency
**Focus**:
- Mixture of Experts scaling to 1000+ experts
- Advanced architectural innovations (MoD, dynamic depth)
- Specialized hardware utilization (TPU-like efficiency on GPU)
- Agentic capabilities and tool use
- Multimodal extension
**Milestones**:
- State-of-the-art performance on standardized benchmarks
- Efficient deployment and serving solutions
- Research publications on architectural innovations

## Suggested Experiments
Based on the architecture, the following experiments would yield valuable insights:

### Ablation Studies
1. **MLA vs Standard Attention**: Compare memory/quality tradeoffs
2. **GDN vs Mamba-2**: Evaluate SSM alternatives
3. **MoE Ablations**: Vary expert count, activation factor, group size
4. **Schedule Variations**: Test different MLA:GDN ratios (3:1, 6:1, 8:1, etc.)
5. **MTP Depth**: Experiment with depth=1,2,3,4,5
6. **μP Impact**: Compare with and without μP re-initialization
7. **Optimizer Comparison**: Muon vs NorMuon vs pure AdamW
8. **Checkpointing Strategies**: Different shard_keep_last values

### Scaling Experiments
1. **Weak Scaling**: Fixed problem size per GPU, increase GPU count
2. **Strong Scaling**: Fixed total problem size, increase GPU count
3. **Parameter Scaling**: Vary model dimension while keeping architecture constant
4. **Expert Scaling**: Increase n_routed_experts while holding activated constant

### Architecture Exploration
1. **Layer Schedule Search**: Automated search for optimal MLA:GDN patterns
2. **Expert Organization**: Different grouping strategies for MoE
3. **Positional Encoding**: Compare RoPE with ALiBi, mixed approaches
4. **Activation Functions**: Experiment with different FFN activations
5. **Normalization**: Compare RMSNorm with LayerNorm, other variants

### System Optimization
1. **Kernel Benchmarking**: Profile and optimize custom kernels
2. **Communication Optimization**: Improve FSDP all-gather patterns
3. **Memory Profiling**: Identify and address memory hotspots
4. **Load Balancing**: Improve expert routing to reduce imbalance
5. **Pipeline Optimization**: Overlap communication with computation

## Risk Mitigation in Roadmap
Each phase includes risk mitigation strategies:

### Technical Risk Mitigation
- **Prototype First**: Implement features in isolation before integration
- **Feature Flags**: New features controlled by config flags
- **Backward Compatibility**: Ensure changes don't break existing workflows
- **Comprehensive Testing**: Unit tests for new components
- **Gradual Rollout**: Enable features for subset of steps/layers initially

### Resource Risk Mitigation
- **Smoke Test First**: Validate changes with smoke_pretrain.yaml
- **Resource Monitoring**: Track memory usage, throughput, stability
- **Checkpoint Frequently**: Enable recovery from failures
- **Deterministic Seeding**: Ensure reproducibility for debugging
- **Incremental Scaling**: Validate at smaller scales before full runs

### Scientific Risk Mitigation
- **Baseline Comparisons**: Always compare to known working configurations
- **Ablation Studies**: Isolate variables in experiments
- **Statistical Significance**: Run multiple seeds for critical comparisons
- **Publication Quality**: Aim for reproducible, publishable results
- **Community Feedback**: Share findings for external validation

## Dependencies and Prerequisites
Progress along this roadmap depends on:

### Technical Dependencies
- **PyTorch Version**: FSDP2 tracking, BF16 support
- **CUDA Architecture**: Ampere/Hopper for BF16, Tensor Cores
- **Hardware Availability**: Access to sufficient GPU resources for scaling
- **Software Dependencies**: lm-eval-harness, W&B, MLflow, etc.

### Knowledge Dependencies
- **Architecture Understanding**: Deep knowledge of current implementation
- **Distributed Systems**: Expertise in FSDP, NCCL, parallelism strategies
- **Optimization Theory**: Understanding of optimizers, schedulers, quantization
- **Systems Performance**: Profiling, bottleneck analysis, optimization

### Resource Dependencies
- **Compute Access**: GPU clusters for experiments at various scales
- **Storage**: Sufficient storage for checkpoints, datasets, logs
- **Network**: Adequate bandwidth for distributed experiments
- **Personnel**: Expert time for implementation and experimentation

## Success Metrics
Progress will be measured by:

### Technical Metrics
- **Training Stability**: Percentage of steps without NaN/Inf
- **Convergence Speed**: Loss decrease per token seen
- **Memory Efficiency**: GB per parameter trained
- **Throughput**: Tokens/second per GPU/watt
- **Checkpoint Reliability**: Success rate of save/resume operations

### Scientific Metrics
- **Benchmark Performance**: Performance on standard NLP benchmarks
- **Scaling Laws**: adherence to expected scaling curves
- **Ablation Insights**: Clear understanding of component contributions
- **Reproducibility**: Consistent results across seeds and runs
- **Novelty**: Contributions to architectural understanding

### Engineering Metrics
- **Code Quality**: Test coverage, maintainability, technical debt
- **Documentation**: Completeness and accuracy of documentation
- **Usability**: Ease of use for new researchers and engineers
- **Integration**: Smooth operation with existing tools and workflows
- **Innovation**: Novel solutions to identified problems

## Conclusion
This research roadmap outlines a path forward for advancing the hybrid architecture implemented in this repository. By focusing on stability, scalability, and scientific rigor, the project can continue to push the boundaries of efficient language model development while maintaining a strong foundation for experimentation and innovation.