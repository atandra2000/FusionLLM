# FusionLLM Architecture Audit Report

**Date**: 2026-06-06  
**Auditor**: Principal Research Engineer  
**Scope**: FusionLLM V2 Stabilization and Architecture Refactor

---

## Executive Summary

FusionLLM is a research-oriented hybrid Transformer model with **MLA + Gated DeltaNet + MoE + MTP** components, implemented with **FSDP2** distributed training. The codebase shows strong engineering quality in core components but has several critical issues requiring immediate attention before large-scale training:

### Critical Issues Identified:
1. **Sequential recurrence in Gated DeltaNet** (lines 173-212 in `gated_deltanet.py`) - the pure PyTorch fallback implementation uses Python loops which will be extremely slow
2. **Triton dependency without fallback path** - some implementations require Triton but fail without it
3. **Dead code paths** - many features are partially implemented without full integration
4. **Documentation vs. Implementation gaps** - several config options lack active code paths

### Strengths:
- Clean MLA implementation with GQA on top of low-rank KV (DeepSeek-V3 style)
- Well-structured distributed training with FSDP2
- Comprehensive checkpoint system with async saving
- μP initialization and NorMuon optimizer for stability

---

## 1. Directory Structure Mapping

```
FusionLLM/
├── config/              # YAML configurations (pretrain.yaml, smoke_pretrain.yaml)
├── data/                # Data loading utilities
│   ├── async_loader.py    # Two-stage async sharded loader
│   ├── curriculum.py      # Curriculum learning scheduler
│   ├── prepare_data.py    # Data preparation utilities
│   └── shard_writer.py    # Binary shard format writer
├── docs/                # Comprehensive documentation (12 markdown files)
├── eval/                # Evaluation utilities
│   ├── eval_core.py       # Perplexity evaluation
│   └── run_lm_eval.py     # lm-eval-harness wrapper
├── kernels/             # Custom kernels
│   ├── flash_attn.py      # FlashAttention 3 wrapper
│   ├── delta_rule.py      # Triton delta-rule kernel
│   ├── ce_softcap.py      # Soft-capped cross-entropy
│   └── linear_relu2.py    # Fused Linear+ReLU²
├── models/              # Model implementations
│   ├── mamba.py           # Legacy Mamba-2
│   ├── mole.py            # MoLE (Mixture of Low-rank Experts)
│   ├── moe.py             # DeepSeek-V3 style MoE
│   ├── mtp.py             # Multi-Token Prediction
│   ├── mup.py             # μP initialization
│   ├── mla.py             # Multi-Head Latent Attention
│   ├── gated_deltanet.py  # Gated DeltaNet (Qwen3-Next style)
│   ├── rope.py            # Shared RoPE implementation
│   └── transformer.py     # Main Transformer backbone
├── ops/triton/          # Triton operations
│   └── grouped_gemm.py    # Grouped GEMM for MoE
├── scripts/             # Launch scripts (smoke, 8xA100)
├── tests/               # Test suite (20+ tests)
├── training/            # Training infrastructure
│   ├── pretrain.py        # Main training loop (FSDP2)
│   ├── normuon.py         # NorMuon optimizer
│   ├── schedules.py       # Batch/seq_len schedules
│   ├── wsd.py             # WSD scheduler
│   └── wsd.py             # Warmup-Stable-Decay scheduler
├── utils/               # Utilities
│   ├── checkpoint.py      # Async checkpoint manager
│   ├── distributed.py     # FSDP2 wrappers and collectives
│   └── logging.py         # W&B + MLflow integration
└── training/pretrain.py # Entry point
```

---

## 2. MLA Implementation Status

### File: `/Users/atandrabharati/Desktop/llm/FusionLLM/models/mla.py`

**Implementation Status: ⭐️ PRODUCTION READY**

**Key Features:**
- Multi-Head Latent Attention with GQA on top of low-rank KV
- QK-norm always on for training stability
- Sliding window option for local attention (Gemma 2 pattern)
- Cached wkv_b projections (absorption trick)
- QK normalization for numerical stability
- RoPE with YaRN scaling support

**Key Classes/Functions:**
```python
MultiHeadLatentAttention(
    config: dict,           # dim, n_heads, q_lora_rank, kv_lora_rank, etc.
    layer_idx: int = 0,
    world_size: int = 1,    # For distributed training
    rank: int = 0
)
```

**Architecture:**
- Query projection: `wq_a/wq_b` (with RMSNorm) or direct `wq`
- KV projection: `wkv_a` (projects to kv_lora_rank + qk_rope_head_dim), `wkv_b` (projects to qk_nope + v)
- RoPE applied to both queries and keys
- FlashAttention 3 integration with fallback to PyTorch SDPA
- Per-GQA group weights

**Strengths:**
- Well-optimized implementation with caching
- GQA reduces KV cache memory by ~8× vs MHA
- Local attention with global attention interleaving (5:1 pattern)

**Issues:**
- None critical — the implementation is solid

---

## 3. Gated DeltaNet Implementation Status

### File: `/Users/atandrabharati/Desktop/llm/FusionLLM/models/gated_deltanet.py`

**Implementation Status: ⚠️ PARTIAL - CRITICAL ISSUE IDENTIFIED**

**Key Classes/Functions:**
```python
GatedDeltaNet(
    config: dict,           # dim, gdn_d_state, gdn_d_conv, gdn_headdim
    layer_idx: int = 0,
    world_size: int = 1,
    rank: int = 0
)

Mamba2Block(
    config: dict,
    layer_idx: int = 0,
    world_size: int = 1,
    rank: int = 0
)
```

**Architecture:**
- 6-stream projection: `(z, x, b, c, dt, g)`
- Causal conv1d (kernel=4) over value stream
- Δ-rule recurrence with state update:
  - `h_t = g_t · h_{t-1} + k_t v_t^T` (outer product)
  - `k_t = B_t / ||B_t||_2` (normalized keys)
  - `y_t = C_t · h_t`
- SwiGLU output gate

**Critical Issue #1: Sequential Recurrence**
```python
# lines 173-212: Reference pure PyTorch implementation
def _delta_rule(self, v, dt, A, B, C) -> torch.Tensor:
    # ...
    for t in range(seqlen):  # ❌ SERIAL TOKEN LOOP - extremely slow
        k_t = F.normalize(B[:, t].to(torch.float32), dim=-1, eps=1e-6)
        v_t = v[:, t].to(torch.float32)
        state = decay[:, t].unsqueeze(-2) * state + v_t.unsqueeze(-2) * k_t.unsqueeze(-2)
        # ...
```

This implementation uses a Python for-loop over sequence length, which will be **extremely slow** for long sequences (4K-16K tokens). This is explicitly flagged in the instruction.md as unacceptable:

> "Any implementation containing `for t in range(seq_len)` inside the critical forward path is unacceptable."

**Critical Issue #2: Triton Dependency**
```python
# lines 142-162
if getattr(self, "use_triton_delta_rule", True):
    from kernels.delta_rule import chunked_delta_rule, has_triton
    if has_triton():
        y = chunked_delta_rule(v, dt, A, B, C)
    else:
        raise RuntimeError("Triton is required for GatedDeltaNet but not available.")
```

There is **no fallback** when Triton is not available. This creates an unacceptable production dependency.

**Potential Fixes Required:**
1. Replace sequential `_delta_rule` with chunked parallel implementation using parallel scan
2. Implement a native PyTorch chunked implementation as fallback
3. Ensure the chunked delta rule kernel (`kernels/delta_rule.py`) is production-ready

**Triton Kernel Status** (`/Users/atandrabharati/Desktop/llm/FusionLLM/kernels/delta_rule.py`):
- Chunked kernel is implemented but has issues in the combine phase (lines 288-321)
- Sequential combine loop in `_CHUNK` increments while processing chunks — not fully optimized
- Requires testing for numerical stability

---

## 4. MoE Routing Implementation Status

### File: `/Users/atandrabharati/Desktop/llm/FusionLLM/models/moe.py`

**Implementation Status: ⭐️ PRODUCTION READY (with minor optimizations)**

**Key Classes/Functions:**
```python
AuxLossFreeGate(config: dict)
DeepSeekMoE(
    config: dict,
    world_size: int = 1,
    rank: int = 0,
    tp_size: int = 1,     # Tensor parallelism
    tp_rank: int = 0
)
```

**Architecture:**
- Top-4 routing (configurable via `n_activated_experts`)
- Shared experts + routed experts
- Group-limited routing: Divide experts into groups, select top-k groups, then top experts within groups
- Aux-loss-free bias-based load balancing

**Routing Mechanism:**
```python
# Biased score: sigmoid(x @ W) + bias
# Routing weights computed from raw sigmoid scores (not biased)
# Bias only affects routing decision, not gradient signal
```

**Expert Setup:**
- Configurable per-instance activation: SwiGLU or ReLU²
- Tensor parallelism support within experts
- Each expert has `w1`, `w2`, `w3` (SwiGLU) or `w1`, `w2` (ReLU²)

**Performance Optimizations:**
- Precomputed expert weight stacks (`_expert_w1_stack`, `_expert_w2_stack`, etc.)
- Scatter-gather routing with capacity factor limits
- Triton grouped-GEMM support (when available)
- All-to-all dispatch option (DeepSeek-V3 style)

**Critical Issue #3: Incomplete All-to-All Implementation**
```python
# lines 468-471, 646-723
if self.use_all_to_all and self.world_size > 1 and dist.is_initialized():
    y_routed = self._all_to_all_dispatch(...)
else:
    # Fall back to scatter-gather
```

The `_all_to_all_dispatch` method (lines 646-723) still uses scatter-gather internally — it's not a full all-to-all implementation. The method comment says:
> "Full implementation would: 1. All-to-all: dispatch tokens to expert ranks... 2. Local expert computation... 3. All-to-all: gather results back..."

But currently only falls back to the scatter-gather path, making the "all-to-all" feature essentially incomplete.

**Recommendations:**
1. Complete the all-to-all implementation or remove the option if not critical
2. Consider using `torch.distributed.all_to_all` primitive for true expert parallelism

---

## 5. MTP Implementation Status

### File: `/Users/atandrabharati/Desktop/llm/FusionLLM/models/mtp.py`

**Implementation Status: ⭐️ PRODUCTION READY**

**Key Classes/Functions:**
```python
MultiTokenPrediction(main_model: nn.Module, config: dict)
MTPBlock(dim: int, n_heads: int, inter_dim: int, is_aux: bool = False)
MTPModule(dim: int, n_heads: int, inter_dim: int, depth: int)
```

**Architecture:**
- `mtp_depth` auxiliary heads (default 3)
- Each head predicts tokens at offset `[depth]` from the current position
- MTP layers fuse previous hidden state with target embedding
- Tied to main model's embed and head

**Loss Function:**
```python
def softcap_ce(logits, target, cap=15.0):
    # loss = cap * tanh(loss_raw / cap)
    # Prevents outlier tokens from dominating training
```

**Weight Schedule:**
- Default: `[0.3, 0.2, 0.1]` for depths 1, 2, 3
- Linear decay for deeper predictions

**Strengths:**
- Clean alignment of logits and targets
- Soft-capped loss for stability
- Tied parameters to avoid duplication

---

## 6. Distributed Training Setup (FSDP2)

### File: `/Users/atandrabharati/Desktop/llm/FusionLLM/utils/distributed.py`

**Implementation Status: ⭐️ PRODUCTION READY**

**Key Functions:**
```python
setup_distributed(backend="nccl") -> Tuple[int, int, int]
cleanup_distributed()
wrap_fsdp2(
    model,
    param_dtype=torch.bfloat16,
    fsdp_shard_strategy="FULL_SHARD",
    fsdp_backward_prefetch=True,
    limit_all_gathers=True
)
configure_reshard(model, keep_last_n=1)
```

**FSDP2 Configuration:**
- **Auto-wrap policy**: Per-TransformerBlock
- **Sharding strategy**: FULL_SHARD (default), SHARD_GRAD_OP, NO_SHARD
- **Backward prefetch**: ON (backward PRE)
- **Forward prefetch**: OFF (saves H2D bandwidth)
- **All-gather limit**: ON

**Distributed Features:**
- NCCL backend for GPU collectives
- BF16 reduced precision for FSDP2 all-gathers
- All-to-all helpers: `all_to_all_single`, `all_to_all`
- Async checkpointing (inCheckpointManager)

**Expert Parallelism:**
- Experts are split across ranks via `DeepSeekMoE`
- World size division: `n_local_experts = n_routed_experts // world_size`
- All-reduce only on routed expert output (shared experts computed locally)

**Strengths:**
- Clean FSDP2 wrapping with auto-wrap policy
- Async checkpointing with background thread
- Memory efficient with reshard tuning

---

## 7. Optimizer Configuration

### Files:
- `/Users/atandrabharati/Desktop/llm/FusionLLM/training/normuon.py`
- `/Users/atandrabharati/Desktop/llm/FusionLLM/configs/pretrain.yaml`
- `/Users/atandrabharati/Desktop/llm/FusionLLM/training/wsd.py`

### Optimizer Setup (from config/pretrain.yaml):
```yaml
optimizer: normuon_adamw       # normuon_adamw | muon_adamw
lr: 3e-4                       # AdamW base LR
muon_lr: 0.02                  # Muon LR (matrix params)
muon_momentum: 0.95
beta: [0.9, 0.95]
weight_decay: 0.1
cautious_wd: true
grad_clip: 1.0
```

### Optimizers:
1. **NorMuon** (`training/normuon.py`): Orthogonalized Adam with per-row RMS
2. **Muon** (if `muon_lr` used): Orthogonalized SGD with Newton-Schulz
3. **AdamW** (for non-matrix params): Standard AdamW with weight decay

### Scheduler: WSD (Warmup-Stable-Decay)
```python
class WSDScheduler:
    - Warmup: linear from 0 to peak over warmup_frac (0.01)
    - Stable: constant at peak for stable_frac (0.84)
    - Decay: linear/cosine decay to min_lr_ratio (0.1)
```

**Instruction.md Requirement:**
> "_muon_lr: 0.02 — Muon LR (matrix params)_" — This suggests muon_adamw optimizer should be used for matrix parameters.

**Current State:**
- The training loop in `pretrain.py` uses NorMuon for all parameters (unless muon_lr is set)
- The config specifies `muon_lr: 0.02` but there's no implementation of a Muon-based optimizer in the codebase
- This may indicate an incomplete implementation or missing file

**Critical Issue #4: Muon Optimizer Missing**
The config references `muon_lr` but a Muon optimizer implementation is not found. This may cause:
1. Errors if user sets `optimizer: muon_adamw`, training will crash
2. Suboptimal training if NorMuon is being used instead of Muon for matrix params

**Note**: Looking at `training/pretrain.py`, I found that Muon IS implemented (lines 329-411), but it's used conditionally based on `cfg.optimizer`. The config `muon_lr: 0.02` is only used when `optimizer == "muon_adamw"`, otherwise NorMuon is used with `lr=oc.muon_lr`.

---

## 8. Block Architecture Summary

### Recommended Block Pattern (from instruction.md):
```
MLA
MLA
MLA
SSM (Gated DeltaNet or Mamba-2)
MLA
MoE
```

**Current Implementation:**
- Layer schedule: `"5:1"` or `"6:1"` — 5 MLA + 1 GDN/Mamba per cycle
- Total layers: 30 (configurable)
- Every 6th layer uses SSM with small dense FFN

### Configuration:
```yaml
layer_schedule: "5:1"   # 5 MLA + 1 GDN
ssm_type: "gdn"         # Gated DeltaNet (default) or "mamba2"
```

---

## 9. Critical Issues Summary

| # | Issue | Severity | affected File(s) | Description |
|---|-------|----------|------------------|-------------|
| 1 | Sequential recurrence in DeltaNet | 🔴 CRITICAL | `models/gated_deltanet.py:173-212` | Python for-loop over seqlen will be extremely slow |
| 2 | Triton dependency without fallback | 🔴 CRITICAL | `models/gated_deltanet.py:153-162` | Fails if Triton not available; no PyTorch fallback |
| 3 | All-to-all MoE dispatch incomplete | 🟡 HIGH | `models/moe.py:646-723` | Falls back to scatter-gather; not true all-to-all |
| 4 | Muon optimizer missing | 🟢 LOW | `training/pretrain.py:329-411` | Muon IS implemented, used conditionally based on config |
| 5 | Incomplete MTP implementation | 🟡 MEDIUM | `models/mtp.py:142` | MTP block uses `proj_aux` for depth ≥ 2 but may not be used |
| 6 | MoLE module incomplete | 🟢 LOW | `models/mole.py` | Referenced in code but no implementation found |

### Detailed Issue Explanations:

#### Issue #1: Sequential Recurrence (HIGHEST PRIORITY)
The `_delta_rule` method in `gated_deltanet.py` (lines 173-212) uses:
```python
for t in range(seqlen):
    # ... token-by-token processing
```

This is **prohibited** by `instruction.md`:
> "Any implementation containing `for t in range(seq_len)` inside the critical forward path is unacceptable."

**Expected impact:**
- 4K tokens: ~4K iterations × batch size
- 16K tokens: ~16K iterations × batch size

For typical batch sizes (2-8) and context lengths (4K-16K), this will cause **catastrophic slowdown**.

#### Issue #2: Triton Dependency
The GatedDeltaNet implementation requires Triton and fails with RuntimeError if not available:
```python
if has_triton():
    y = chunked_delta_rule(v, dt, A, B, C)
else:
    raise RuntimeError("Triton is required for GatedDeltaNet")
```

This is problematic because:
- Triton is not always available (requires CUDA + specific versions)
- The reference implementation (lines 173-212) is too slow to use as production fallback
- Users may install without Triton and hit runtime errors

#### Issue #3: MoE All-to-All Incomplete
The `_all_to_all_dispatch` method (lines 646-723) says:
> "Full implementation would: 1. All-to-all: dispatch tokens to expert ranks..."

But actually falls back to:
```python
# Fall back to optimized scatter-gather for now
active_list = active_indices.tolist()
scatter_gather_needed = True
if len(active_list) > 0 and self._try_grouped_gemm(...):
    scatter_gather_needed = False
# ... more scatter-gather logic
```

The "all-to-all" feature is essentially useless until fully implemented.

#### Issue #4: Muon Optimizer Status
**RESOLVED**: Muon IS implemented in `training/pretrain.py` (lines 329-411) and is used conditionally based on config. The `muon_lr` config key is only used when `optimizer == "muon_adamw"`, otherwise NorMuon uses `lr=oc.muon_lr`.

**Note**: This means the config references `muon_lr` but it's only used with `muon_adamw` optimizer. When using `normuon_adamw` (the default), the `muon_lr` config value is unused and NorMuon uses `lr` instead.

---

## 10. Recommendations

### Phase 1: Fix Critical Issues (1-2 days)

1. **Replace sequential DeltaNet with chunked parallel implementation**
   - Delete or heavily refactor `_delta_rule` method
   - Ensure `chunked_delta_rule` kernel (Kernels/delta_rule.py) is production-ready
   - Add native PyTorch fallback as backup (not Triton-only)

2. **Resolve Muon optimizer discrepancy**
   - The config `muon_lr: 0.02` is only used when `optimizer == "muon_adamw"`
   - Default optimizer is `normuon_adamw` which uses `lr: 3e-4` for matrix params
   - **No action needed** - the optimizer implementation matches config intent

3. **Complete or remove All-to-All MoE dispatch**
   - Implement proper `torch.distributed.all_to_all` for expert parallelism
   - Or: Remove the option and document that scatter-gather is the only dispatcher

### Phase 2: Testing & Validation (2-3 days)

4. **Run smoke tests on 8×A100**
   - Ensure distributed training works
   - Verify checkpoint save/load cycles
   - Test async checkpointing performance

5. **Validate DeltaNet on GPU**
   - Test with synthetic data first
   - Compare chunked vs. reference (if still used)
   - Profile to ensure no sequential bottleneck

### Phase 3: Documentation (1 day)

6. **Document known issues and workarounds**
   - Create `docs/KNOWN_ISSUES.md`
   - Update config descriptions with current state
   - Add migration guide from old to new implementation

### Phase 4: Production Readiness (1 week)

7. **Stress test at scale**
   - Run 1000 steps on 8×A100
   - Monitor memory usage
   - Check for gradient instabilities
   - Validate checkpoint recovery

---

## 11. Final Status: Architecture Audit Conclusion

**Overall Architecture Quality: ⭐ 7.5/10**

### Strengths:
- Clean MLA implementation (production-ready)
- Well-structured distributed training (FSDP2)
- Good checkpoint system with async support
- Comprehensive documentation
- μP initialization and normalization strategies
- Multi-Token Prediction with soft-capped loss

### Critical Issues (Must Fix Before Production):
1. **Sequential DeltaNet** — Prohibited by architecture requirements
2. **Triton dependency** — No fallback path for non-Triton systems
3. **All-to-all MoE** — Incomplete implementation

### Code Quality:
- ⭐ High: MLA implementation, distributed training, checkpoint system
- ⚠️ Medium: DeltaNet implementation, MoE all-to-all
- ⚠️ Low: Test coverage (20+ tests but needs more edge cases)

### Deployment Readiness:
- **Smoketest**: Ready (but with known DeltaNet performance issues)
- **8×A100 production**: **NOT READY** — requires fixes to DeltaNet and optimizer
- **Long-context training (16K+)**: NOT READY — sequential DeltaNet will be too slow

---

## Next Steps

1. Fix DeltaNet sequential recurrence (highest priority)
2. Add PyTorch fallback when Triton unavailable
3. Complete or remove All-to-All MoE
4. Run comprehensive smoke tests
5. Profile DeltaNet performance before and after fixes

Do not proceed to large-scale training until these critical issues are resolved.
