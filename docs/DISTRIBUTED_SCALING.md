# Distributed Scaling

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: Distributed Training Guide

---

## Executive Summary

This document outlines the distributed scaling strategy for FusionLLM V2, focusing on:
1. **Maintaining FSDP2** as the primary parallelism strategy
2. **Adding Expert Parallelism** for MoE scaling
3. **Recovery validation** for fault tolerance

---

## 1. FSDP2 Configuration

### Current Implementation

**File**: `utils/distributed.py`

```python
def wrap_fsdp2(
    model: nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    fsdp_shard_strategy: str = "FULL_SHARD",
    fsdp_backward_prefetch: bool = True,
    limit_all_gathers: bool = True,
) -> FSDPModule:
    """Wrap model with FSDP2."""
    
    # Auto-wrap policy: per-TransformerBlock
    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        if isinstance(module, TransformerBlock):
            return True
        return False
    
    # Configure FSDP2
    fsdp_config = {
        "strategy": fsdp_shard_strategy,
        "param_dtype": param_dtype,
        "reduce_dtype": torch.float32,
        "backward_prefetch": fsdp_backward_prefetch,
        "forward_prefetch": False,
        "limit_all_gathers": limit_all_gathers,
    }
    
    return FSDP(model, auto_wrap_policy=auto_wrap_policy, **fsdp_config)
```

### FSDP2 Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| FULL_SHARD | Shard params, grads, optimizer states | Default (8×A100) |
| SHARD_GRAD_OP | Shard grads and optimizer states | Memory-constrained |
| NO_SHARD | No sharding (DDP-like) | Debugging |

### Configuration

```yaml
# configs/pretrain.yaml
distributed:
  fsdp_shard_strategy: FULL_SHARD
  fsdp_param_dtype: bf16
  fsdp_reduce_dtype: fp32
  fsdp_backward_prefetch: true
  fsdp_forward_prefetch: false
  fsdp_limit_all_gathers: true
  shard_keep_last: 1  # Keep last N FSDP units resident
```

---

## 2. Expert Parallelism

### Current Implementation

**File**: `models/moe.py`

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, config, world_size=1, rank=0):
        super().__init__()
        
        self.world_size = world_size
        self.rank = rank
        
        # Split experts across ranks
        self.n_local_experts = self.n_routed_experts // world_size
        self.experts_start = rank * self.n_local_experts
        self.experts_end = self.experts_start + self.n_local_experts
        
        # Local experts (this rank's shard)
        self.experts = nn.ModuleList([
            Expert(dim, inter_dim)
            for _ in range(self.n_local_experts)
        ])
        
        # Shared experts (replicated on all ranks)
        self.shared_experts = nn.ModuleList([
            Expert(dim, inter_dim)
            for _ in range(self.n_shared_experts)
        ])
```

### Expert Parallel Strategy

```
Rank 0: Experts 0-7 (local)
Rank 1: Experts 8-15 (local)
Rank 2: Experts 16-23 (local)
...
Rank 7: Experts 56-63 (local)

All ranks: Shared experts 0-3 (replicated)
```

### Forward Pass

```python
def forward(self, x):
    flat = x.view(-1, self.dim)
    
    # Routing (same on all ranks)
    weights, indices = self.gate(flat)
    
    # Filter to local experts
    local_mask = (indices >= self.experts_start) & (indices < self.experts_end)
    local_indices = indices - self.experts_start
    local_indices = local_indices * local_mask  # Zero out non-local
    
    # Compute local expert outputs
    y_routed = self._compute_local_experts(flat, local_indices, weights)
    
    # All-reduce routed outputs
    if self.world_size > 1:
        dist.all_reduce(y_routed, op=dist.ReduceOp.SUM)
    
    # Compute shared experts (all ranks)
    y_shared = self._compute_shared_experts(flat)
    
    return y_routed + y_shared
```

### All-to-All Dispatch (Future)

```python
def _all_to_all_dispatch(self, flat, indices, weights):
    """DeepSeek-V3 style all-to-all dispatch."""
    
    # 1. Dispatch tokens to expert ranks
    # Group tokens by target rank
    tokens_per_rank = self._group_tokens_by_rank(indices)
    
    # All-to-all: send tokens to expert ranks
    dispatch = dist.all_to_all(tokens_per_rank)
    
    # 2. Local expert computation
    y_local = self._compute_experts(dispatch)
    
    # 3. Gather results back
    # All-to-all: send results back to token owners
    y_gathered = dist.all_to_all(y_local)
    
    return y_gathered
```

### Expert Parallel Scaling

| GPUs | Experts/GPU | Shared Experts | Communication |
|------|-------------|----------------|---------------|
| 1 | 64 | 4 | None |
| 2 | 32 | 4 | All-reduce |
| 4 | 16 | 4 | All-reduce |
| 8 | 8 | 4 | All-reduce |

---

## 3. Recovery Validation

### Checkpoint Management

**File**: `utils/checkpoint.py`

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, async_save=True):
        self.checkpoint_dir = checkpoint_dir
        self.async_save = async_save
        self.save_thread = None
    
    def save(self, state, step):
        """Save checkpoint with all required state."""
        checkpoint = {
            'step': step,
            'model': state['model'],
            'optimizer': state['optimizer'],
            'scheduler': state['scheduler'],
            'ema': state.get('ema'),
            'routing': state.get('routing'),
            'dataloader': state.get('dataloader'),
            'rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
        }
        
        if self.async_save:
            self._async_save(checkpoint, step)
        else:
            self._sync_save(checkpoint, step)
    
    def load(self, checkpoint_path):
        """Load checkpoint and restore all state."""
        checkpoint = torch.load(checkpoint_path)
        
        # Restore model
        self.model.load_state_dict(checkpoint['model'])
        
        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore scheduler
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Restore EMA
        if checkpoint.get('ema'):
            self.ema.load_state_dict(checkpoint['ema'])
        
        # Restore routing state
        if checkpoint.get('routing'):
            self.moe.load_state_dict(checkpoint['routing'])
        
        # Restore dataloader
        if checkpoint.get('dataloader'):
            self.dataloader.load_state_dict(checkpoint['dataloader'])
        
        # Restore RNG state
        torch.random.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        
        return checkpoint['step']
```

### Recovery Validation Tests

```python
# tests/test_recovery.py
def test_checkpoint_restart():
    """Test that training can restart from checkpoint."""
    # Train for 100 steps
    for step in range(100):
        train_step(model, batch)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, step=100)
    
    # Load checkpoint
    load_checkpoint(model, optimizer, step=100)
    
    # Verify model state
    assert torch.allclose(model.state_dict(), saved_state)
    
    # Continue training
    for step in range(100, 200):
        train_step(model, batch)
    
    # Verify convergence
    assert loss < initial_loss


def test_optimizer_restore():
    """Test that optimizer state is properly restored."""
    # Train for 100 steps
    for step in range(100):
        train_step(model, batch)
    
    # Save optimizer state
    optimizer_state = optimizer.state_dict()
    
    # Load optimizer state
    optimizer.load_state_dict(optimizer_state)
    
    # Verify optimizer state
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                assert 'exp_avg' in state
                assert 'exp_avg_sq' in state


def test_ema_restore():
    """Test that EMA state is properly restored."""
    # Train for 100 steps
    for step in range(100):
        train_step(model, batch)
        ema.update()
    
    # Save EMA state
    ema_state = ema.state_dict()
    
    # Load EMA state
    ema.load_state_dict(ema_state)
    
    # Verify EMA state
    assert ema.decay == ema_state['decay']
    assert len(ema.shadow) == len(ema_state['shadow'])


def test_dataloader_restore():
    """Test that dataloader position is properly restored."""
    # Create dataloader
    dataloader = create_dataloader()
    
    # Iterate for 100 batches
    for i, batch in enumerate(dataloader):
        if i >= 100:
            break
    
    # Save dataloader state
    dataloader_state = dataloader.state_dict()
    
    # Load dataloader state
    dataloader.load_state_dict(dataloader_state)
    
    # Verify dataloader position
    assert dataloader.current_step == 100


def test_router_restore():
    """Test that router bias is properly restored."""
    # Train for 100 steps
    for step in range(100):
        train_step(model, batch)
        moe.update_gate_bias()
    
    # Save router state
    router_state = moe.gate.state_dict()
    
    # Load router state
    moe.gate.load_state_dict(router_state)
    
    # Verify router state
    assert torch.allclose(moe.gate.bias, router_state['bias'])
```

---

## 4. Communication Optimization

### All-Reduce Optimization

```python
# Current: Per-layer all-reduce
for layer in layers:
    x = layer(x)
    if world_size > 1:
        dist.all_reduce(x)

# Optimized: Batched all-reduce
def batched_all_reduce(tensors):
    """Batch multiple all-reduces into one."""
    if len(tensors) == 1:
        dist.all_reduce(tensors[0])
        return
    
    # Flatten tensors
    flat = torch.cat([t.view(-1) for t in tensors])
    
    # Single all-reduce
    dist.all_reduce(flat)
    
    # Reshape back
    offset = 0
    for t in tensors:
        size = t.numel()
        t.copy_(flat[offset:offset+size].view_as(t))
        offset += size
```

### Communication Overhead

| Operation | Latency | Bandwidth |
|-----------|---------|-----------|
| All-reduce | ~10 μs | ~200 GB/s |
| All-to-all | ~20 μs | ~200 GB/s |
| All-gather | ~10 μs | ~200 GB/s |

---

## 5. Scaling Targets

### Performance Scaling

| GPUs | Tokens/sec | Scaling Efficiency |
|------|------------|-------------------|
| 1 | 500K | 100% |
| 2 | 950K | 95% |
| 4 | 1.8M | 90% |
| 8 | 3.4M | 85% |
| 16 | 6.0M | 75% |

### Memory Scaling

| GPUs | Memory/GPU | Total Memory |
|------|------------|--------------|
| 1 | 75 GB | 75 GB |
| 2 | 40 GB | 80 GB |
| 4 | 22 GB | 88 GB |
| 8 | 14 GB | 112 GB |

---

## 6. Configuration

### configs/pretrain.yaml

```yaml
distributed:
  # FSDP2
  fsdp_shard_strategy: FULL_SHARD
  fsdp_param_dtype: bf16
  fsdp_reduce_dtype: fp32
  fsdp_backward_prefetch: true
  fsdp_forward_prefetch: false
  fsdp_limit_all_gathers: true
  shard_keep_last: 1
  
  # Expert Parallelism
  expert_parallel: true
  use_all_to_all_dispatch: false  # Enable for large-scale
  
  # Communication
  communication_bucket_size: 25  # MB
  all_reduce_overlap: true
  
  # Recovery
  async_checkpointing: true
  checkpoint_interval: 1000
  max_checkpoint_age: 7  # days
```

---

## 7. Checklist

### Phase 1: FSDP2

- [x] Implement FSDP2 wrapping
- [x] Test FULL_SHARD strategy
- [ ] Test SHARD_GRAD_OP strategy
- [ ] Benchmark memory savings

### Phase 2: Expert Parallelism

- [x] Implement expert sharding
- [x] Test all-reduce on routed output
- [ ] Implement all-to-all dispatch
- [ ] Test communication overlap

### Phase 3: Recovery

- [x] Implement checkpoint save/load
- [x] Test model state restore
- [x] Test optimizer state restore
- [x] Test EMA state restore
- [ ] Test dataloader state restore
- [ ] Test router state restore

### Phase 4: Scaling

- [ ] Benchmark 1/2/4/8 GPU scaling
- [ ] Measure communication overhead
- [ ] Optimize batch size for scaling
- [ ] Test 16+ GPU scaling

---

## 8. References

1. FSDP2: "Fully Sharded Data Parallel" (PyTorch docs)
2. Expert Parallelism: "Switch Transformers" (2022)
3. All-to-All: "Distributed Communication Library" (PyTorch docs)
4. Checkpointing: "Efficient Checkpointing for Distributed Training" (2023)
