# Migration Plan

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: Step-by-Step Migration Guide

---

## Executive Summary

This document provides a step-by-step migration plan from FusionLLM V1 to V2, covering:
1. **Code changes** required for compatibility
2. **Configuration updates** for new features
3. **Testing and validation** steps
4. **Rollback procedures** if issues arise

---

## 1. Migration Overview

### What's Changing

| Component | V1 | V2 | Impact |
|-----------|----|----|--------|
| Gated DeltaNet | Sequential loop | Chunked parallel | High |
| MoE Routing | Softmax | Biased sigmoid | Medium |
| MTP Targets | t+1/t+2/t+3 | t+1/t+2/t+4 | Low |
| FSDP | DDP/FSDP1 | FSDP2 | High |
| Optimizer | AdamW | NorMuon + CautiousAdamW | Medium |

### Migration Priority

1. **Critical**: Gated DeltaNet (sequential → chunked)
2. **High**: FSDP2 migration
3. **Medium**: MoE routing changes
4. **Low**: MTP target updates

---

## 2. Pre-Migration Checklist

### Backup

```bash
# 1. Backup current code
cp -r FusionLLM FusionLLM_V1_backup

# 2. Backup checkpoints
cp -r checkpoints checkpoints_V1_backup

# 3. Backup configs
cp -r configs configs_V1_backup

# 4. Export current state
git stash
git checkout -b v2-migration
```

### Environment

```bash
# 1. Create new environment
conda create -n fusionllm_v2 python=3.10
conda activate fusionllm_v2

# 2. Install dependencies
pip install torch>=2.4.0
pip install triton>=3.2.0  # Optional, for Triton kernels
pip install flash-attn>=3.0.0
pip install wandb mlflow

# 3. Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Step-by-Step Migration

### Step 1: Update Gated DeltaNet (CRITICAL)

**File**: `models/gated_deltanet.py`

**Changes**:
1. Replace sequential `_delta_rule` method
2. Add PyTorch fallback for Triton
3. Update forward pass to use chunked implementation

**Verification**:
```bash
# Run delta-rule tests
python -m pytest tests/test_delta_rule.py -v

# Verify no sequential loop
grep -n "for t in range(seqlen)" models/gated_deltanet.py
# Should return no matches
```

### Step 2: Update MoE Routing

**File**: `models/moe.py`

**Changes**:
1. Replace softmax with biased sigmoid routing
2. Add group-limited routing
3. Add shared experts

**Verification**:
```bash
# Run MoE tests
python -m pytest tests/test_moe.py -v

# Verify routing mechanism
python -c "from models.moe import AuxLossFreeGate; print('OK')"
```

### Step 3: Update MTP Targets

**File**: `models/mtp.py`

**Changes**:
1. Update target offsets from t+1/t+2/t+3 to t+1/t+2/t+4
2. Update loss weights

**Verification**:
```bash
# Run MTP tests
python -m pytest tests/test_mtp.py -v
```

### Step 4: Migrate to FSDP2

**File**: `utils/distributed.py`

**Changes**:
1. Replace FSDP1 with FSDP2
2. Update wrapping logic
3. Add expert parallelism support

**Verification**:
```bash
# Run distributed tests
python -m pytest tests/test_distributed.py -v

# Test FSDP2 wrapping
python -c "from utils.distributed import wrap_fsdp2; print('OK')"
```

### Step 5: Update Optimizer

**File**: `training/normuon.py`, `training/pretrain.py`

**Changes**:
1. Add NorMuon optimizer
2. Add CautiousAdamW optimizer
3. Update training loop

**Verification**:
```bash
# Run optimizer tests
python -m pytest tests/test_optimizer.py -v

# Test NorMuon
python -c "from training.normuon import NorMuon; print('OK')"
```

### Step 6: Update Configuration

**File**: `configs/pretrain.yaml`

**Changes**:
```yaml
# V1 Configuration
model:
  ssm_type: "mamba2"
  n_routed_experts: 32
  mtp_depth: 2

# V2 Configuration
model:
  ssm_type: "gdn"
  n_routed_experts: 64
  n_shared_experts: 4
  mtp_depth: 3

training:
  # Add FSDP2 settings
  fsdp_shard_strategy: FULL_SHARD
  fsdp_param_dtype: bf16
  
  # Add NorMuon settings
  optimizer: normuon_adamw
  muon_lr: 0.02
```

---

## 4. Testing and Validation

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_mla.py -v
python -m pytest tests/test_moe.py -v
python -m pytest tests/test_delta_rule.py -v
python -m pytest tests/test_mtp.py -v
python -m pytest tests/test_distributed.py -v
```

### Smoke Test

```bash
# Run smoke test on 1 GPU
python training/pretrain.py --config configs/smoke_pretrain.yaml

# Run smoke test on 8 GPUs
torchrun --nproc_per_node=8 training/pretrain.py --config configs/smoke_pretrain.yaml
```

### Benchmark Test

```bash
# Run performance benchmarks
python -m benchmarks.benchmark_delta_rule --seqlen 4096
python -m benchmarks.benchmark_moe --n_experts 64
python -m benchmarks.benchmark_training --steps 100
```

### Convergence Test

```bash
# Train for 1000 steps and verify convergence
python training/pretrain.py --config configs/pretrain.yaml --max_steps 1000

# Check loss curve
tensorboard --logdir runs/
```

---

## 5. Validation Checklist

### Code Validation

- [ ] No sequential loops in critical path
- [ ] All tests passing
- [ ] No deprecated warnings
- [ ] Type hints complete

### Performance Validation

- [ ] DeltaNet speedup > 10×
- [ ] MoE routing overhead < 10%
- [ ] Memory usage within budget
- [ ] Throughput > 3.5M tokens/sec

### Stability Validation

- [ ] No NaN/Inf in training
- [ ] Gradient norms < 10.0
- [ ] Expert utilization > 0.1
- [ ] Routing entropy > 0.5

### Convergence Validation

- [ ] Loss decreasing
- [ ] Perplexity < 15.0 after 1000 steps
- [ ] No loss spikes
- [ ] Stable expert routing

---

## 6. Rollback Procedures

### Code Rollback

```bash
# If issues arise, rollback to V1
git checkout v1-stable

# Or restore from backup
cp -r FusionLLM_V1_backup FusionLLM
```

### Checkpoint Rollback

```bash
# If checkpoint issues, restore from backup
cp -r checkpoints_V1_backup/* checkpoints/
```

### Configuration Rollback

```bash
# If config issues, restore V1 config
cp configs_V1_backup/pretrain.yaml configs/pretrain.yaml
```

---

## 7. Migration Timeline

### Week 1: Preparation

- [ ] Backup all code and checkpoints
- [ ] Set up new environment
- [ ] Review all changes

### Week 2: Core Migration

- [ ] Update Gated DeltaNet
- [ ] Update MoE routing
- [ ] Update MTP targets

### Week 3: Infrastructure

- [ ] Migrate to FSDP2
- [ ] Update optimizer
- [ ] Update configuration

### Week 4: Validation

- [ ] Run all tests
- [ ] Run smoke tests
- [ ] Run benchmarks
- [ ] Validate convergence

### Week 5: Production

- [ ] Deploy to production
- [ ] Monitor training
- [ ] Address any issues

---

## 8. Common Issues and Solutions

### Issue 1: Triton Not Available

**Symptom**: RuntimeError in GatedDeltaNet

**Solution**: PyTorch fallback is now available. No action required.

### Issue 2: FSDP2 Wrapping Errors

**Symptom**: RuntimeError in wrap_fsdp2

**Solution**: Ensure model is wrapped correctly:
```python
from utils.distributed import wrap_fsdp2
model = wrap_fsdp2(model, param_dtype=torch.bfloat16)
```

### Issue 3: MoE Routing Divergence

**Symptom**: Loss spikes, expert collapse

**Solution**: Adjust router warmup:
```yaml
model:
  moe_warmup_steps: 2000
  router_initial_temperature: 2.0
```

### Issue 4: Memory Overflow

**Symptom**: CUDA out of memory

**Solution**: Enable activation checkpointing:
```yaml
training:
  use_checkpoint: true
  checkpoint_policy: selective
```

---

## 9. Post-Migration Tasks

### Documentation

- [ ] Update README.md
- [ ] Update API documentation
- [ ] Create migration guide for users
- [ ] Update benchmarks

### Monitoring

- [ ] Set up W&B dashboard
- [ ] Set up MLflow tracking
- [ ] Configure alerts
- [ ] Monitor training stability

### Support

- [ ] Create FAQ
- [ ] Set up support channel
- [ ] Document known issues
- [ ] Create troubleshooting guide

---

## 10. Contact

For migration support:
- **Technical Lead**: [Name]
- **Slack Channel**: #fusionllm-v2-migration
- **Email**: support@fusionllm.dev

---

## 11. References

1. FSDP2 Migration: "FSDP2 Migration Guide" (PyTorch docs)
2. Mamba-2: "Mamba-2: Linear Sequence Modeling" (2024)
3. DeepSeek-V3: "DeepSeek-V3 Technical Report" (2024)
4. NorMuon: "modded-nanogpt speedrun" (2024)
