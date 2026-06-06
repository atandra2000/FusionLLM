# FusionLLM V2 Architecture

**Version**: 2.0  
**Date**: 2026-06-06  
**Status**: Architecture Specification

---

## Executive Summary

FusionLLM V2 is a hybrid Transformer architecture combining:
- **Multi-Head Latent Attention (MLA)** - dominant modeling mechanism
- **Gated DeltaNet (GDN)** - linear-complexity state-space layer
- **DeepSeek-V3 MoE** - sparse expert routing with load balancing
- **Multi-Token Prediction (MTP)** - auxiliary training objective

The design prioritizes training stability, throughput, and convergence over architectural novelty.

---

## Core Design Principles

1. **Stability First**: Every component must pass numerical stability tests
2. **Throughput**: Target 4M+ tokens/sec on 8×A100 SXM
3. **Simplicity**: Avoid complex routing, hierarchical structures, or fragile optimizations
4. **Scalability**: Must scale to 16K context with FSDP2

---

## Block Architecture

### Repeating Pattern (5 MLA + 1 GDN)

```
Layer 0: MLA + MoE FFN
Layer 1: MLA + MoE FFN
Layer 2: MLA + MoE FFN
Layer 3: MLA + MoE FFN
Layer 4: MLA + MoE FFN
Layer 5: GDN + Dense FFN
```

This pattern repeats 6 times for a total of 36 layers (6 × 6).

### Layer Type Distribution

| Type | Count | Depth % | Purpose |
|------|-------|---------|---------|
| MLA | 30 | 83% | Primary attention mechanism |
| GDN | 6 | 17% | Linear-complexity state-space |

---

## Component Specifications

### 1. Multi-Head Latent Attention (MLA)

**Status**: Production-ready  
**File**: `models/mla.py`

#### Architecture
- **Query**: Low-rank projection via `wq_a → RMSNorm → wq_b`
- **KV**: Low-rank projection via `wkv_a → (kv_lora, rope_dim) → wkv_b`
- **GQA**: 8 KV groups shared by 32 Q heads (4 Q per group)
- **QK-norm**: Always enabled for training stability
- **Sliding window**: 2048 tokens (local attention) with 5:1 global interleaving

#### Hyperparameters
```yaml
n_heads: 32
n_kv_groups: 8
q_lora_rank: 512
kv_lora_rank: 256
qk_nope_head_dim: 128
qk_rope_head_dim: 64
v_head_dim: 128
sliding_window: 2048
```

#### Key Features
- **KV cache absorption**: `wkv_b` cached for efficient inference
- **RoPE**: Applied to both Q and K with YaRN scaling (factor=8.0)
- **FlashAttention 3**: Primary backend with PyTorch SDPA fallback
- **Per-GQA group weights**: Learnable attention scaling

#### Memory Efficiency
- KV cache: `kv_lora_rank + qk_rope_head_dim` = 320 per token per layer
- ~8× reduction vs MHA (would be 2048 per token)

---

### 2. Gated DeltaNet (GDN)

**Status**: Production-ready (with PyTorch fallback)  
**File**: `models/gated_deltanet.py`

#### Architecture
```
Input → in_proj → (z, x, b, c, dt, g)
         ↓
    conv1d(x) → SiLU
         ↓
    B, C, dt, g projections
         ↓
    Δ-rule recurrence (chunked parallel)
         ↓
    y + v * D (skip connection)
         ↓
    y * g * SiLU(z) (output gate)
         ↓
    out_proj
```

#### Key Features
- **6-stream projection**: `(z, x, b, c, dt, g)`
- **Causal conv1d**: Kernel size 4, depth-wise
- **Δ-rule recurrence**: State update with forget factor
- **Chunked parallel processing**: No sequential token loop
- **PyTorch fallback**: Works without Triton

#### Hyperparameters
```yaml
gdn_d_state: 128
gdn_d_conv: 4
gdn_headdim: 64
```

#### Performance
- **Chunk size**: 64 tokens (configurable)
- **Complexity**: O(seqlen × n_chunks) vs O(seqlen²) for attention
- **Triton kernel**: Available for CUDA systems
- **PyTorch fallback**: Available for all systems

---

### 3. DeepSeek-V3 MoE

**Status**: Production-ready  
**File**: `models/moe.py`

#### Architecture
```
Input → gate (biased sigmoid) → top-k routing
         ↓
    Shared experts (unconditional)
         ↓
    Routed experts (conditional)
         ↓
    Merge (shared + routed)
```

#### Key Features
- **Aux-loss-free load balancing**: Bias-based routing
- **Group-limited routing**: 8 groups, top-3 groups per token
- **Shared experts**: 4 unconditional experts
- **Routed experts**: 64 experts, 6 active per token
- **Expert dropout**: 10% during warmup for diversity

#### Hyperparameters
```yaml
n_routed_experts: 64
n_shared_experts: 4
n_activated_experts: 6
n_expert_groups: 8
n_limited_groups: 3
group_topk: 2
expert_capacity_factor: 1.5
```

#### Routing Mechanism
```python
biased_score = sigmoid(x @ W) + bias  # routing decision
weights = raw_sigmoid_scores[selected]  # weight computation
weights = weights / weights.sum()  # normalize
```

#### Load Balancing
- **Bias update**: After each optimizer step
- **Threshold**: ±10% from average
- **Speed**: 0.001 per step

---

### 4. Multi-Token Prediction (MTP)

**Status**: Production-ready  
**File**: `models/mtp.py`

#### Architecture
```
Hidden state → MTP block → predict t+1, t+2, t+4
```

#### Key Features
- **Depth**: 3 auxiliary heads
- **Targets**: t+1, t+2, t+4 (not t+8)
- **Loss**: Soft-capped cross-entropy (cap=15.0)
- **Weight schedule**: [0.3, 0.2, 0.1]
- **Tied parameters**: Embedding and head shared with main model

#### Loss Function
```python
def softcap_ce(logits, target, cap=15.0):
    raw_loss = F.cross_entropy(logits, target)
    return cap * torch.tanh(raw_loss / cap)
```

---

### 5. Dense FFN (GDN layers)

**Status**: Production-ready  
**File**: `models/moe.py` (Expert class)

#### Architecture
```
Input → Linear → SwiGLU → Linear → Output
```

#### Hyperparameters
```yaml
inter_dim: 4096
ffn_activation: swiglu
```

---

## Training Configuration

### Optimizer

**Primary**: NorMuon (matrix params) + CautiousAdamW (rest)

```yaml
optimizer: normuon_adamw
lr: 3e-4              # AdamW (non-matrix)
muon_lr: 0.02         # NorMuon (matrix)
weight_decay: 0.1
cautious_wd: true
```

### Scheduler

**WSD**: Warmup-Stable-Decay

```yaml
scheduler: wsd
warmup_frac: 0.01
stable_frac: 0.84
decay: linear
```

### Batch/Sequence Schedule

```yaml
micro_batch_size: 2
gradient_accumulation_steps: 16
max_seq_len: 4096
```

---

## Distributed Training

### FSDP2 Configuration

```yaml
shard_strategy: FULL_SHARD
param_dtype: bf16
reduce_dtype: fp32
backward_prefetch: true
forward_prefetch: false
limit_all_gathers: true
```

### Expert Parallelism

- Experts split across ranks: `n_local_experts = 64 // world_size`
- All-reduce only on routed expert output
- Shared experts computed locally on all ranks

---

## Memory Budget

### Static State (per GPU, 8×A100 80GB)

| Component | Size |
|-----------|------|
| MLA params | ~1.2 GB |
| GDN params | ~0.3 GB |
| MoE params | ~2.5 GB |
| MTP params | ~0.1 GB |
| Optimizer state | ~1.5 GB |
| **Total** | ~5.6 GB |

### Activation Memory (per micro-batch)

| Context | MLA | GDN | MoE | Total |
|---------|-----|-----|-----|-------|
| 4096 | 2.1 GB | 0.3 GB | 1.2 GB | 3.6 GB |
| 8192 | 4.2 GB | 0.6 GB | 2.4 GB | 7.2 GB |
| 16384 | 8.4 GB | 1.2 GB | 4.8 GB | 14.4 GB |

---

## Numerical Stability

### Safeguards

1. **QK-norm**: Always enabled
2. **BF16 autocast**: Forward pass
3. **FP32 reduction**: Attention softmax, loss computation
4. **Gradient clipping**: Max norm = 1.0
5. **Logit softcap**: 15.0 for main loss, 15.0 for MTP
6. **Router z-loss**: Prevents logits from growing too large

### Initialization

- **μP**: Enabled for transfer learning
- **NorMuon**: Per-row RMS normalization for matrix params
- **Cautious WD**: Sign-masked weight decay

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Tokens/sec (8×A100) | >4M | ~3.5M |
| Memory efficiency | >80% | ~75% |
| Convergence (150B tokens) | <3.0 PPL | TBD |
| Training stability | No NaN/Inf | ✓ |

---

## Migration Notes

### From V1 to V2

1. **Gated DeltaNet**: Replaced sequential loop with chunked parallel
2. **MoE**: Added group-limited routing and shared experts
3. **MTP**: Updated to t+1/t+2/t+4 targets
4. **FSDP2**: Maintained, added expert parallelism

### Breaking Changes

- Config keys renamed for consistency
- MoE routing now uses biased sigmoid (not softmax)
- MTP targets changed from t+1/t+2/t+3 to t+1/t+2/t+4

---

## References

1. Yang et al., "Gated DeltaNet: Sequence Modeling with a Linear Time-Complexity Recurrent Network" (2025)
2. DeepSeek-V3 Technical Report (2024)
3. Keller Jordan, "modded-nanogpt speedrun" (2024)
4. FlashAttention 3 (2024)
5. μP: Tensor Programs V (2023)
