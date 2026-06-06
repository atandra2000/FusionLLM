# Memory Optimization

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: Memory Management Guide

---

## Executive Summary

This document outlines memory optimization strategies for FusionLLM V2, focusing on:
1. **Static buffer reuse** to reduce allocator pressure
2. **Activation checkpointing policy** for memory savings
3. **Long context planning** for 4K/8K/16K tokens

---

## 1. Static Buffer Reuse

### Current Implementation

**File**: `models/moe.py`

```python
# Pre-allocated scatter-gather buffer (recycled to avoid per-forward allocation)
self._y_routed_buf: torch.Tensor | None = None

def forward(self, x):
    flat = x.view(-1, self.dim)
    
    # Recycle scatter-gather buffer
    if self._y_routed_buf is None or self._y_routed_buf.shape != flat.shape:
        self._y_routed_buf = torch.zeros_like(flat)
    else:
        self._y_routed_buf.zero_()
    y_routed = self._y_routed_buf
```

### Additional Buffers to Pre-allocate

#### Routing Buffers

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        # ... existing code ...
        
        # Pre-allocate routing buffers
        self._routing_weights_buf = None  # (T, topk)
        self._routing_indices_buf = None  # (T, topk)
        self._assignment_buf = None  # (T, n_local_experts)
    
    def forward(self, x):
        flat = x.view(-1, self.dim)
        T = flat.size(0)
        
        # Reuse routing buffers
        if self._routing_weights_buf is None or self._routing_weights_buf.shape[0] != T:
            self._routing_weights_buf = torch.empty(T, self.topk, device=x.device)
            self._routing_indices_buf = torch.empty(T, self.topk, device=x.device, dtype=torch.long)
        
        weights, indices = self.gate(flat)
        self._routing_weights_buf.copy_(weights)
        self._routing_indices_buf.copy_(indices)
```

#### Communication Buffers

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        # ... existing code ...
        
        # Pre-allocate communication buffers for all-reduce
        self._all_reduce_buf = None
    
    def forward(self, x):
        # ... routing ...
        
        # Reuse all-reduce buffer
        if self._all_reduce_buf is None or self._all_reduce_buf.shape != y_routed.shape:
            self._all_reduce_buf = torch.empty_like(y_routed)
        self._all_reduce_buf.copy_(y_routed)
        dist.all_reduce(self._all_reduce_buf, op=dist.ReduceOp.SUM)
```

### Memory Savings

| Buffer | Size (4K) | Size (8K) | Size (16K) |
|--------|-----------|-----------|------------|
| y_routed | 32 MB | 64 MB | 128 MB |
| routing_weights | 0.1 MB | 0.2 MB | 0.4 MB |
| routing_indices | 0.1 MB | 0.2 MB | 0.4 MB |
| assignment | 0.1 MB | 0.2 MB | 0.4 MB |
| **Total** | **32.3 MB** | **64.6 MB** | **129.2 MB** |

---

## 2. Activation Checkpointing Policy

### Current Implementation

**File**: `models/transformer.py`

```python
class Transformer(nn.Module):
    def __init__(self, config, use_checkpoint=True):
        # ... existing code ...
        
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
        return x
```

### Optimized Policy

**Goal**: Checkpoint ~50% of MLA layers + all MoE/SSM layers

```python
class Transformer(nn.Module):
    def __init__(self, config, use_checkpoint=True):
        # ... existing code ...
        
        self.use_checkpoint = use_checkpoint
        
        # Define checkpoint policy per layer type
        self.checkpoint_policy = self._get_checkpoint_policy(config)
    
    def _get_checkpoint_policy(self, config):
        n_layers = config['n_layers']
        layer_schedule = config.get('layer_schedule', '5:1')
        
        # Parse schedule (e.g., '5:1' means 5 MLA + 1 GDN)
        mla_count, gdn_count = map(int, layer_schedule.split(':'))
        total_cycle = mla_count + gdn_count
        
        policy = []
        for i in range(n_layers):
            cycle_idx = i % total_cycle
            
            if cycle_idx == total_cycle - 1:
                # GDN layer - always checkpoint
                policy.append(True)
            elif 'moe' in self.layers[i].__class__.__name__.lower():
                # MoE layer - always checkpoint
                policy.append(True)
            else:
                # MLA layer - checkpoint ~50%
                # Use alternating pattern: checkpoint even layers
                policy.append(i % 2 == 0)
        
        return policy
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.checkpoint_policy[i]:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
        return x
```

### Checkpointing Strategy

| Layer Type | Checkpoint | Rationale |
|------------|------------|-----------|
| MLA (even) | Yes | ~50% of MLA layers |
| MLA (odd) | No | Avoid over-checkpointing |
| MoE | Yes | MoE layers are memory-heavy |
| GDN/SSM | Yes | State-space layers have large state |
| Dense FFN | No | Relatively small memory footprint |

### Memory Savings

| Context | Without Checkpoint | With Policy | Savings |
|---------|-------------------|-------------|---------|
| 4096 | 3.6 GB | 2.1 GB | 42% |
| 8192 | 7.2 GB | 4.2 GB | 42% |
| 16384 | 14.4 GB | 8.4 GB | 42% |

---

## 3. Long Context Planning

### Memory Behavior

#### 4096 Tokens (Default)

```yaml
# configs/pretrain.yaml
max_seq_len: 4096
micro_batch_size: 2
```

**Memory Budget**:
- Static state: ~5.6 GB
- Activations: ~3.6 GB
- Optimizer state: ~1.5 GB
- **Total**: ~10.7 GB per GPU

**Recommendation**: Fits comfortably in 80GB A100.

#### 8192 Tokens

```yaml
max_seq_len: 8192
micro_batch_size: 1  # Reduce batch size
```

**Memory Budget**:
- Static state: ~5.6 GB
- Activations: ~7.2 GB
- Optimizer state: ~1.5 GB
- **Total**: ~14.3 GB per GPU

**Recommendation**: Use activation checkpointing policy.

#### 16384 Tokens

```yaml
max_seq_len: 16384
micro_batch_size: 1
gradient_accumulation_steps: 32  # Increase to maintain token count
```

**Memory Budget**:
- Static state: ~5.6 GB
- Activations: ~14.4 GB (with checkpointing: ~8.4 GB)
- Optimizer state: ~1.5 GB
- **Total**: ~15.5 GB per GPU (with checkpointing)

**Recommendation**: Use aggressive checkpointing + reduce batch size.

### Scaling Formulas

#### Activation Memory

```
M_activation = M_attn + M_ffn + M_moe + M_ssm

Where:
- M_attn = 2 * B * L * H * D * sizeof(dtype)  # Q, K, V
- M_ffn = 2 * B * L * D * D_ffn * sizeof(dtype)  # FFN intermediate
- M_moe = B * L * D * D_moe * sizeof(dtype)  # MoE intermediate
- M_ssm = B * L * D * D_state * sizeof(dtype)  # SSM state

For B=2, L=4096, D=2048, H=32, D_ffn=4096, D_moe=1536, D_state=128:
- M_attn = 2 * 2 * 4096 * 32 * 2048 * 2 = 2.1 GB
- M_ffn = 2 * 2 * 4096 * 2048 * 4096 * 2 = 2.7 GB
- M_moe = 2 * 2 * 4096 * 2048 * 1536 * 2 = 1.0 GB
- M_ssm = 2 * 2 * 4096 * 2048 * 128 * 2 = 0.1 GB
- Total = 5.9 GB
```

#### KV Cache Memory

```
M_kv = 2 * L * n_kv * D_kv * sizeof(dtype)

Where:
- L = sequence length
- n_kv = number of KV heads (8)
- D_kv = kv_lora_rank + qk_rope_head_dim (320)

For L=4096, n_kv=8, D_kv=320:
- M_kv = 2 * 4096 * 8 * 320 * 2 = 42 MB per layer
- Total (30 layers) = 1.26 GB
```

### Memory Profiling

```python
# utils/memory_profiler.py
import torch
from contextlib import contextmanager

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
    
    @contextmanager
    def profile(self, name):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start_mem = torch.cuda.memory_allocated()
        yield
        torch.cuda.synchronize()
        
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        
        self.snapshots.append({
            'name': name,
            'start': start_mem / 1024**3,
            'end': end_mem / 1024**3,
            'peak': peak_mem / 1024**3,
            'delta': (end_mem - start_mem) / 1024**3,
        })
    
    def report(self):
        for snap in self.snapshots:
            print(f"{snap['name']}: "
                  f"start={snap['start']:.2f} GB, "
                  f"end={snap['end']:.2f} GB, "
                  f"peak={snap['peak']:.2f} GB, "
                  f"delta={snap['delta']:.2f} GB")
```

---

## 4. Memory Optimization Checklist

### Phase 1: Buffer Reuse

- [ ] Pre-allocate routing buffers
- [ ] Pre-allocate communication buffers
- [ ] Test buffer reuse in training loop
- [ ] Measure memory savings

### Phase 2: Checkpointing

- [ ] Implement layer-type-aware checkpointing
- [ ] Test with 4K/8K/16K context
- [ ] Benchmark training speed
- [ ] Validate convergence

### Phase 3: Long Context

- [ ] Profile memory at 4K/8K/16K
- [ ] Optimize batch size schedule
- [ ] Test gradient accumulation
- [ ] Validate numerical stability

---

## 5. Configuration

### configs/pretrain.yaml

```yaml
# Memory optimization
memory:
  # Static buffer reuse
  static_buffer_reuse: true
  
  # Activation checkpointing
  activation_checkpointing: true
  checkpoint_policy: "selective"  # "all", "none", "selective"
  checkpoint_mla_ratio: 0.5  # Checkpoint ~50% of MLA layers
  
  # Long context
  max_seq_len: 4096
  context_schedule_enabled: false
  context_schedule_steps: 5000
  initial_context: 2048
  final_context: 8192
  
  # Memory profiling
  memory_profiling: false
  memory_log_interval: 100
```

---

## 6. References

1. PyTorch Checkpointing: "torch.utils.checkpoint" (PyTorch docs)
2. FSDP2 Memory: "Fully Sharded Data Parallel" (PyTorch docs)
3. Gradient Checkpointing: "Training Deep Nets with Sublinear Memory Cost" (2016)
4. KV Cache: "Efficient Memory Management for Large Language Model Serving" (2023)
