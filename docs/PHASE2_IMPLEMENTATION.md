# Phase 2 Implementation Plan

**Date**: 2026-06-06  
**Status**: In Progress  
**Branch**: v2-stabilization

---

## Executive Summary

Phase 2 focuses on production readiness, inference optimization, and training stability enhancements. This document outlines the implementation plan for achieving a production-ready pretraining codebase.

---

## Phase 2.1: Production Hardening (Week 1)

### 2.1.1 Numerical Health Checks

**Objective**: Prevent silent training failures through comprehensive monitoring.

**Implementation**:

1. **Loss Spike Detection** (`training/numerical_health.py`)
   - Rolling window statistics (EMA, moving percentiles)
   - Configurable alert thresholds (z-score, absolute delta)
   - Automatic checkpoint saving on spike detection

2. **Gradient Anomaly Detection**
   - Per-parameter gradient norm tracking
   - Anomaly detection using rolling statistics
   - Optional gradient clipping override

3. **Activation Monitoring**
   - Optional activation statistics collection
   - NaN/Inf detection hooks
   - Memory usage tracking

**Files to Create**:
- `training/numerical_health.py`: Core health monitoring
- `tests/test_numerical_health.py`: Unit tests

### 2.1.2 Loss Computation

**Objective**: Standardize loss computation with proper handling of special tokens.

**Implementation**:

1. **Standardized Cross-Entropy** (`training/loss.py`)
   - Reduction: 'sum' with num_tokens denominator
   - Label smoothing: 0.0 default
   - Z-loss: 0.0 default
   - Optional softcap: 15.0

2. **MTP Auxiliary Loss** (`training/loss.py`)
   - Configurable weight: 0.0 default
   - Per-layer MSE loss
   - Proper masking (ignore first token)

3. **MoE Load-Balancing Loss** (`models/moe.py`)
   - Configurable weight: 0.01 default
   - Load-balancing term
   - Router z-loss term

**Files to Create**:
- `training/loss.py`: Standardized loss functions
- `tests/test_loss.py`: Unit tests

### 2.1.3 Optimizer Integration

**Objective**: Ensure proper loss scaling and optimizer configuration.

**Implementation**:

1. **NorMuon Optimizer** (`training/normuon.py`)
   - Proper gradient clipping (1.0)
   - μP re-initialization
   - WSD schedule with μP warmup

2. **Configuration Validation**
   - Optimizer-specific parameter validation
   - Learning rate schedule validation
   - Warmup/cooldown step validation

**Files to Modify**:
- `training/normuon.py`: Add validation
- `training/optim.py`: Add configuration checks

---

## Phase 2.2: Performance Optimization (Week 2)

### 2.2.1 torch.compile Optimization

**Objective**: Optimize inference performance through compilation.

**Implementation**:

1. **Compilation Strategy** (`utils/compile.py`)
   - Compile only: Attention, FFN, MoE router, SSM layers
   - Skip: Embeddings, Head, RMSNorm, Reshape ops
   - Mode: 'max-autotune' for inference
   - Dynamic shapes: Enabled

2. **Compilation Configuration**
   - `reduce-overhead`: For small tensors
   - `max-autotune`: For large tensors
   - `inductor`: For CPU inference

3. **Verification**:
   - AOT autograd compatibility
   - CUDA graph compatibility
   - Triton kernel compatibility

**Files to Create**:
- `utils/compile.py`: Compilation utilities
- `tests/test_compile.py`: Compilation tests

### 2.2.2 MoE Routing Optimization

**Objective**: Reduce MoE routing overhead through vectorized operations.

**Implementation**:

1. **Vectorized Scatter-Gather** (`models/moe.py`)
   - Replace sequential loop with batch operations
   - Use `torch.scatter_add` for accumulation
   - Optimize buffer reuse

2. **Routing Optimization**
   - Cache routing decisions
   - Reduce memory allocations
   - Optimize sorting operations

3. **Profiling**:
   - Measure routing overhead
   - Compare before/after optimization
   - Document performance gains

**Files to Modify**:
- `models/moe.py`: Vectorized operations
- `benchmarks/benchmark_moe.py`: Performance comparison

### 2.2.3 NCCL Profiling

**Objective**: Identify communication bottlenecks.

**Implementation**:

1. **NCCL Profiler** (`utils/nccl_profiler.py`)
   - NCCL version detection
   - Communication pattern analysis
   - Latency measurement

2. **Profiling Integration**
   - Optional profiling hooks
   - Communication overhead tracking
   - Bandwidth utilization metrics

**Files to Create**:
- `utils/nccl_profiler.py`: NCCL profiling utilities
- `tests/test_nccl_profiler.py`: Unit tests

---

## Phase 2.3: Testing & Validation (Week 3)

### 2.3.1 Smoke Test Suite

**Objective**: Validate core functionality.

**Implementation**:

1. **Forward Pass Tests**
   - Single device
   - Multi-GPU
   - Different batch sizes

2. **Training Step Tests**
   - Single step
   - Gradient accumulation
   - Mixed precision

3. **Checkpoint Tests**
   - Save/load
   - Resumption
   - Consistency

**Files to Create**:
- `tests/test_smoke.py`: Smoke test suite

### 2.3.2 Numerical Stability Tests

**Objective**: Ensure training stability.

**Implementation**:

1. **Gradient Flow Tests**
   - Vanishing gradients
   - Exploding gradients
   - Per-layer gradients

2. **Mixed Precision Tests**
   - FP16/BF16 stability
   - Loss scaling
   - Overflow detection

3. **Long-Running Tests**
   - 1000+ step stability
   - Loss curve consistency
   - Gradient norm trends

**Files to Create**:
- `tests/test_numerical_stability.py`: Stability tests

### 2.3.3 Convergence Validation

**Objective**: Validate training convergence.

**Implementation**:

1. **Baseline Comparison**
   - Reference loss curves
   - Convergence speed metrics
   - Final performance metrics

2. **Metric Tracking**
   - Loss curves
   - Gradient norms
   - Learning rates
   - Memory usage

3. **Reporting**
   - Automated report generation
   - Visual comparison charts
   - Performance regression detection

**Files to Create**:
- `tests/test_convergence.py`: Convergence validation
- `utils/convergence_report.py`: Report generation

---

## Phase 2.4: Documentation & Integration (Week 4)

### 2.4.1 Documentation Updates

**Objective**: Complete documentation for production deployment.

**Implementation**:

1. **API Documentation**
   - Module interfaces
   - Configuration options
   - Usage examples

2. **Performance Guide**
   - Benchmark results
   - Optimization recommendations
   - Hardware requirements

3. **Deployment Guide**
   - Environment setup
   - Configuration management
   - Monitoring setup

**Files to Create**:
- `docs/PHASE2_PERFORMANCE.md`: Performance guide
- `docs/PHASE2_DEPLOYMENT.md`: Deployment guide

### 2.4.2 CI/CD Integration

**Objective**: Automate testing and benchmarking.

**Implementation**:

1. **GitHub Actions Workflow**
   - Unit tests
   - Integration tests
   - Performance benchmarks

2. **Performance Regression Detection**
   - Benchmark comparison
   - Threshold-based alerts
   - Historical tracking

3. **Documentation Generation**
   - Automated API docs
   - Benchmark reports
   - Convergence reports

**Files to Create**:
- `.github/workflows/phase2.yml`: CI/CD workflow

---

## Phase 2.5: Inference Optimization (Optional)

### 2.5.1 KV Cache Optimization

**Objective**: Optimize inference memory usage.

**Implementation**:

1. **PagedAttention** (if applicable)
   - Memory-efficient attention
   - Dynamic memory allocation
   - Reduced memory fragmentation

2. **KV Cache Management**
   - Efficient caching strategies
   - Memory pooling
   - Cache eviction policies

### 2.5.2 Speculative Decoding

**Objective**: Accelerate inference through speculation.

**Implementation**:

1. **Draft Model**
   - Small draft model
   - Verification mechanism
   - Acceptance criteria

2. **Verification**
   - Speculative tokens
   - Rollback mechanism
   - Performance measurement

---

## Implementation Schedule

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Production Hardening | Numerical health, Loss computation, Optimizer fixes |
| 2 | Performance Optimization | torch.compile, MoE optimization, NCCL profiling |
| 3 | Testing & Validation | Smoke tests, Numerical stability, Convergence validation |
| 4 | Documentation & Integration | Documentation, CI/CD, Performance guide |

---

## Success Criteria

### Phase 2.1: Production Hardening
- [ ] Numerical health checks implemented
- [ ] Loss computation standardized
- [ ] Optimizer validation complete

### Phase 2.2: Performance Optimization
- [ ] torch.compile integrated and verified
- [ ] MoE routing optimized
- [ ] NCCL profiling implemented

### Phase 2.3: Testing & Validation
- [ ] Smoke tests passing
- [ ] Numerical stability verified
- [ ] Convergence validated

### Phase 2.4: Documentation & Integration
- [ ] Documentation complete
- [ ] CI/CD workflow operational
- [ ] Performance guide published

---

## Risk Mitigation

### Performance Risks
- **torch.compile Compatibility**: Test with AOT autograd and CUDA graphs
- **MoE Optimization**: Benchmark before/after to ensure no regression
- **NCCL Profiling**: Use non-intrusive profiling to avoid overhead

### Stability Risks
- **Numerical Health**: Implement gradual rollout with monitoring
- **Loss Computation**: Validate with existing checkpoints
- **Optimizer Changes**: Test with existing training runs

### Integration Risks
- **CI/CD**: Start with limited scope, expand gradually
- **Documentation**: Focus on critical paths first
- **Inference**: Prioritize production-ready features

---

## Dependencies

### External Libraries
- PyTorch 2.0+ (for torch.compile)
- Triton (for kernel optimization)
- NCCL (for distributed profiling)

### Internal Dependencies
- Phase 1 optimizations (completed)
- Existing test infrastructure
- Documentation framework

---

## Next Steps

1. **Start with Phase 2.1**: Implement numerical health checks
2. **Parallelize Phase 2.2**: Begin torch.compile integration
3. **Establish testing baseline**: Create smoke test suite
4. **Document progress**: Update implementation plan as needed

---

## References

1. `docs/PHASE1_IMPLEMENTATION.md`: Phase 1 summary
2. `docs/PERFORMANCE_REFACTOR.md`: Performance optimization guide
3. `docs/MEMORY_OPTIMIZATION.md`: Memory optimization strategies
4. `docs/TRAINING_STABILITY_PLAN.md`: Training stability plan
