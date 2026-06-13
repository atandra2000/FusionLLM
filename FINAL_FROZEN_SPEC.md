# FusionLLM-v1 Frozen Architecture Specification

**Status**: FROZEN — IMPLEMENTATION-READY
**Version**: v1.0-final
**Date**: 2026-06-13
**Target Hardware**: Single NVIDIA A100 80GB
**Target Stack**: Pure PyTorch, BF16, Context 4096
**Active Parameters**: ~415.6M
**Total Parameters (all experts)**: ~868.6M
**Chinchilla-Optimal Token Budget**: ~8.31B
**Estimated Wall-Clock**: 5–10 days (midpoint ~5.2 days)

---

## 0. Immutability Notice

This document is the **single source of truth** for FusionLLM-v1. All numerical
values below are hardcoded. No architectural toggles exist in the v1 codebase.
Implementation must match these values to the integer. Any deviation requires
re-opening the architecture review, which is forbidden for v1.

---

## 1. Frozen Model Configuration

```yaml
model:
  # -------- Core dimensions (FROZEN) --------
  vocab_size: 64000
  max_seq_len: 4096
  dim: 768                       # d_model
  n_layers: 24                   # 16 MLA + 8 GDN (every 3rd layer is GDN)

  # -------- Layer schedule (FROZEN) --------
  layer_schedule: "gdn_every_3"  # GDN at indices [2,5,8,11,14,17,20,23]

  # -------- MLA (FROZEN) --------
  n_heads: 12
  n_kv_groups: 8                 # GQA ratio 1.5:1
  q_lora_rank: 192
  kv_lora_rank: 96
  qk_nope_head_dim: 64
  qk_rope_head_dim: 32
  v_head_dim: 64
  rope_theta: 10000.0
  rope_factor: 1.0
  qk_norm: true
  sliding_window: null           # disabled for v1

  # -------- MoE (FROZEN — REDUCED FROM ORIGINAL 3072) --------
  n_routed_experts: 8
  n_shared_experts: 1
  n_activated_experts: 2         # top-2
  moe_inter_dim: 2048            # 2.67x dim (REDUCED from 3072 for stability)
  expert_capacity_factor: 1.5
  expert_dropout_prob: 0.0
  moe_warmup_steps: 0
  moe_activation: "swiglu"
  n_expert_groups: 1
  route_scale: 1.0
  bias_upper_threshold: 0.10
  bias_lower_threshold: 0.10
  gate_bias_enabled: true        # aux-loss-free biased sigmoid

  # -------- GDN (FROZEN — REDUCED FROM ORIGINAL 1536/64) --------
  ssm_type: "gdn"
  gdn_d_state: 32                # REDUCED from 64
  gdn_d_conv: 4
  gdn_headdim: 32
  gdn_d_inner: 1024              # REDUCED from 1536
  gdn_n_heads: 32                # = d_inner / headdim
  gdn_chunk_size: 64             # pure-PyTorch chunk size for delta-rule
  gdn_use_triton: false          # FORBIDDEN in v1

  # -------- Dense FFN (GDN layers) (FROZEN — REDUCED) --------
  inter_dim: 2048                # 2.67x dim (REDUCED from 3072)
  ffn_activation: "swiglu"

  # -------- MTP (FROZEN — LOSS WEIGHTS REDUCED) --------
  mtp_depth: 2
  mtp_loss_weight_1: 0.10        # REDUCED from 0.30
  mtp_loss_weight_2: 0.05        # REDUCED from 0.20
  mtp_softcap: true
  mtp_softcap_value: 15.0
  mtp_n_heads: 12
  mtp_inter_dim: 2048            # matches FFN
  mtp_share_attention: true      # MTP uses shared MLA/FFN block
  mtp_tied_head: true

  # -------- μP Initialization --------
  muP: true

  # -------- Logit processing --------
  logit_softcap: 15.0
  asymmetric_rescale: false

  # -------- Embedding / Head --------
  tie_embeddings: true
  no_bias_linear: true           # all Linear layers bias-free except MoE gate
```

---

## 2. Frozen Training Configuration

```yaml
training:
  # -------- Batch strategy (FROZEN) --------
  micro_batch_size: 2            # per step (RECOMMENDED, see Section 7)
  gradient_accumulation_steps: 16
  # Effective batch = 2 * 16 = 32 sequences
  # Tokens per step = 32 * 4096 = 131,072

  # -------- Step budget (FROZEN) --------
  total_tokens: 8_312_928_000    # ~20x active_params (Chinchilla-optimal)
  total_steps: 63_400            # ceil(8.31B / 131,072)
  warmup_steps: 634              # 1% of total
  cooldown_start_frac: 0.84      # WSD schedule

  # -------- Optimizer (FROZEN) --------
  optimizer: "normuon_cautious_adamw"
  lr: 3.0e-4                     # AdamW base LR
  muon_lr: 0.02                  # NorMuon LR
  muon_momentum: 0.95
  adamw_betas: [0.9, 0.95]
  min_lr_ratio: 0.1
  weight_decay: 0.1
  cautious_wd: true
  grad_clip: 1.0
  muon_param_filter: "mlp_and_gdn"  # Muon for 2D matrices in MLP/GDN; AdamW for rest

  # -------- Precision (FROZEN) --------
  dtype: bf16
  use_checkpoint: true           # gradient checkpointing on all 24 layers
  use_triton: false              # FORBIDDEN
  use_flash_attention: false     # FORBIDDEN
  use_sdpa: true                 # PyTorch native SDPA only
  matmul_precision: "high"       # torch.set_float32_matmul_precision

  # -------- MoE balance (FROZEN) --------
  balance_loss_alpha: 1.0e-4
  bias_update_speed: 1.0e-3
  bias_update_every: 10
  z_loss_alpha: 0.0              # disabled for v1

  # -------- Scheduler (FROZEN) --------
  scheduler: "wsd"
  wsd_warmup_frac: 0.01
  wsd_stable_frac: 0.84
  wsd_decay: "linear"

  # -------- Numerical health --------
  loss_spike_threshold: 3.0
  grad_norm_threshold: 10.0
  loss_nan_skip: true

  # -------- Checkpointing --------
  save_dir: "checkpoints/pretrain"
  save_interval_steps: 2000
  save_max_keep: 3
  log_interval_steps: 50

  # -------- Evaluation --------
  eval_enabled: true
  eval_interval_steps: 5000
  eval_max_batches: 8
  eval_synthetic: true

  # -------- Logging --------
  wandb_enabled: true
  wandb_project: "fusionllm-v1"
  wandb_tags: ["v1-frozen", "single-gpu", "a100-80gb", "pure-pytorch"]
```

---

## 3. Frozen Hardware Configuration

```yaml
hardware:
  device: "cuda"
  profile: "a100_80gb_1x"
  min_vram_gb: 70.0              # refuse to start if < 70 GB free
  enable_tf32: true
  enable_bf16_reduced_precision: true
  cudnn_benchmark: true
  cudnn_deterministic: false
  num_workers: 4
  prefetch_factor: 2
  empty_cache_every: 100
  async_checkpointing: false     # synchronous save
  async_wandb: true
  use_mmap_data: true
```

---

## 4. Frozen Data Configuration

```yaml
data:
  shard_manifest_path: "data/shards/manifest.jsonl"
  tokenizer: "bpe_64k"           # 64,000 vocab BPE
  data_mix:
    fineweb_edu: 0.60
    stack_edu: 0.20
    openr1_math: 0.10
    cosmopedia: 0.10
  packing: true                  # sequence packing to 4096
  shuffle_buffer: 10_000
  num_workers: 4
```

---

## 5. Frozen Architecture Details

### 5.1 Layer Structure (24 layers)

| Index | Type        | FFN/Expert Block          |
|-------|-------------|---------------------------|
| 0     | MLA         | MoE (8 routed + 1 shared) |
| 1     | MLA         | MoE (8 routed + 1 shared) |
| 2     | GDN         | Dense FFN                 |
| 3     | MLA         | MoE (8 routed + 1 shared) |
| 4     | MLA         | MoE (8 routed + 1 shared) |
| 5     | GDN         | Dense FFN                 |
| 6     | MLA         | MoE (8 routed + 1 shared) |
| 7     | MLA         | MoE (8 routed + 1 shared) |
| 8     | GDN         | Dense FFN                 |
| 9     | MLA         | MoE (8 routed + 1 shared) |
| 10    | MLA         | MoE (8 routed + 1 shared) |
| 11    | GDN         | Dense FFN                 |
| 12    | MLA         | MoE (8 routed + 1 shared) |
| 13    | MLA         | MoE (8 routed + 1 shared) |
| 14    | GDN         | Dense FFN                 |
| 15    | MLA         | MoE (8 routed + 1 shared) |
| 16    | MLA         | MoE (8 routed + 1 shared) |
| 17    | GDN         | Dense FFN                 |
| 18    | MLA         | MoE (8 routed + 1 shared) |
| 19    | MLA         | MoE (8 routed + 1 shared) |
| 20    | GDN         | Dense FFN                 |
| 21    | MLA         | MoE (8 routed + 1 shared) |
| 22    | MLA         | MoE (8 routed + 1 shared) |
| 23    | GDN         | Dense FFN                 |

**Counts**: 16 MLA layers, 8 GDN layers (every 3rd starting at index 2).

### 5.2 MLA Block (FROZEN, unchanged from v1)

```
Input (B, T, 768)
  │
  ├─ wq_a: Linear(768→192) + RMSNorm(192) + wq_b: Linear(192→12×96=1152)
  │         └─ Split: Q_nope (12, 64), Q_pe (12, 32) → RoPE(Q_pe)
  │
  ├─ wkv_a: Linear(768→128) → Split: KV_latent (96), K_pe (32) → RoPE(K_pe)
  │         └─ RMSNorm(96) → wkv_b: Linear(96→8×128=1024)
  │                    └─ Split: K_nope (8, 64), V (8, 64)
  │
  ├─ Absorption: Q_nope @ wkv_b_k  →  (B, T, 12, 64)
  ├─ GQA expand K/V: 8 → 12 groups
  ├─ Concat: Q = [Q_nope_proj, Q_pe], K = [K_nope, K_pe]
  ├─ QK-Norm: RMSNorm on Q and K (decoupled head_dim=96)
  ├─ SDPA (causal, no FA, no Triton) with math/efficient backend
  └─ wo: Linear(768→768)
```

**Per-layer params**: 1,155,616 (~1.16M)

### 5.3 MoE Block (FROZEN, moe_inter_dim REDUCED 3072→2048)

```
Input (T, 768)
  │
  ├─ Gate: Linear(768→8) + bias → Sigmoid + bias → Top-2
  │         (aux-loss-free biased sigmoid routing)
  │
  ├─ 8 Routed experts (SwiGLU, 768→2048→768):
  │     y = SiLU(W1·x) ⊙ (W3·x);  out = W2·y
  │     Active per token: top-2 of 8
  │
  ├─ 1 Shared expert (SwiGLU, 768→2048→768): always active
  │
  └─ Output = Σ(weight_i × expert_i(x)) + shared_expert(x)
```

**Per-layer active params**: ~14.16M
**Per-layer total params (all 9 experts)**: ~42.47M
**×16 layers active**: ~226.59M
**×16 layers total**: ~679.58M

**Routing**: Aux-loss-free biased sigmoid (DeepSeek-V3 style). Bias terms updated every 10 steps with speed 1e-3. No auxiliary balance loss required, but `balance_loss_alpha=1e-4` is kept as a safety floor.

### 5.4 GDN Block (FROZEN, d_inner REDUCED 1536→1024, d_state REDUCED 64→32)

```
Input (B, T, 768)
  │
  ├─ in_proj: Linear(768→6×1024=6144) → Split: z, x, b, c, dt, g  (each 1024)
  │
  ├─ x → Conv1d(1024, k=4, groups=1024, causal) → SiLU → x_conv
  │
  ├─ b_proj: Linear(1024→32×32=1024)  →  B (B, T, 32, 32)
  ├─ c_proj: Linear(1024→32×32=1024)  →  C (B, T, 32, 32)
  ├─ dt_proj: Linear(1024→32) + dt_bias → SoftPlus → dt
  ├─ g_proj: Linear(1024→1024) → Sigmoid → g
  │
  ├─ v = x_conv.view(B, T, 32, 32)     # headdim = 32
  │
  ├─ A = -exp(A_log)   # (32, 32) fixed decay
  │
  ├─ Delta-rule recurrence (chunked, pure PyTorch, chunk_size=64):
  │     for each chunk of 64 tokens:
  │         state = decay * state + k_chunk^T @ v_chunk      # write
  │         y_chunk = c_chunk @ state + D * v_chunk          # read+skip
  │
  ├─ y = y * g * SiLU(z)   # gating
  └─ out_proj: Linear(1024→768)
```

**Per-layer params**: ~8.69M
**×8 layers**: ~69.51M

**Critical constraint**: Delta-rule must be implemented in pure PyTorch with explicit chunking. No `torch.compile` of the inner loop is allowed in v1 (forbidden by "no Triton" extension). Chunk size 64 is the largest that fits comfortably in BF16 register pressure without CUDA kernel fusion.

### 5.5 Dense FFN (GDN layers, FROZEN, inter_dim REDUCED 3072→2048)

```
Input (T, 768)
  │
  ├─ w1: Linear(768→2048)   # gate
  ├─ w3: Linear(768→2048)   # up
  ├─ w2: Linear(2048→768)   # down
  └─ y = w2(SiLU(w1·x) ⊙ (w3·x))
```

**Per-layer params**: ~4.72M
**×8 layers**: ~37.75M

### 5.6 MTP Block (FROZEN, depth=2, loss weights REDUCED)

```
Main model forward → main_logits, main_hidden (B, T, 768)

MTP Module 1 (depth=1):
  ├─ proj: Linear(2×768→768)              # 1,179,648 params
  │     input = concat(main_hidden[t], embed[t+1])
  ├─ SharedTransformerBlock (pre-norm, MLA, dense FFN)
  │     - MLA: same as v1 (1,155,616 params)
  │     - FFN: dense 768→2048→768 (4,718,592 params)
  ├─ norm + tied output head → logits_1
  └─ Target: tokens[t+2]
  Loss_1 = softcap_CE(logits_1, tokens[t+2], cap=15.0)
  Weight_1 = 0.10

MTP Module 2 (depth=2):
  ├─ proj_aux: Linear(2×768→768)          # 1,179,648 params
  │     input = concat(hidden_1[t], embed[t+2])
  ├─ SharedTransformerBlock (pre-norm, MLA, dense FFN)
  ├─ norm + tied output head → logits_2
  └─ Target: tokens[t+3]
  Loss_2 = softcap_CE(logits_2, tokens[t+3], cap=15.0)
  Weight_2 = 0.05
```

**Total MTP params**: ~14.11M (including 2 shared attn+FFN blocks)

**Final loss**:
```
L = L_main + 0.10 * L_mtp1 + 0.05 * L_mtp2
```

---

## 6. Frozen Numerical Recipe

| Setting                 | Value           | Rationale                                            |
|-------------------------|-----------------|------------------------------------------------------|
| Init scheme             | μP (DeepSeek)   | Stable across width/depth scaling                    |
| Gate init               | Zero bias       | Allows routing entropy to emerge from gradient       |
| A_log init              | log(uniform(1, 16)) | Standard GDN initialization                      |
| dt_bias init            | Uniform(0.001, 0.1)  | Standard                                              |
| D init                  | Ones            | Skip connection strength                            |
| Embedding init          | N(0, 0.02)      | Standard                                             |
| Output head             | Tied            | Halves embedding-related params                      |
| QK-Norm                 | Enabled         | Stabilizes attention logits                          |
| Logit softcap           | 15.0            | Prevents logit explosion in MTP heads                |
| Softcap formula         | `cap * tanh(x/cap)` | GELU-equivalent saturation                       |
| BF16 reduced precision  | Enabled         | Allows intermediate matmul to use TF32-equivalent    |
| Gradient clipping       | L2, max=1.0     | Per-step norm clip                                   |
| Loss spike threshold    | 3.0 × EMA       | Auto-rollback on divergence                          |

---

## 7. Frozen Batch Strategy

**Recommended**: `micro_batch_size=2`, `gradient_accumulation_steps=16`

| Setting | Value | Justification |
|---------|-------|---------------|
| micro_batch_size | 2 | Best SDPA utilization on A100; ~10.6 GB peak VRAM |
| gradient_accumulation_steps | 16 | Effective batch = 32 sequences = 131,072 tokens |
| tokens_per_step | 131,072 | Standard Chinchilla scaling batch |
| total_steps | 63,400 | 8.31B tokens / 131,072 |
| warmup_steps | 634 | 1% of total |
| cooldown_start_step | 53,256 | 84% of total (WSD schedule) |

**Alternatives allowed** (all fit in 80GB):
| micro_batch | grad_accum | tokens/step | VRAM (est.) | Use case |
|-------------|------------|-------------|-------------|----------|
| 1 | 32 | 131,072 | ~10.3 GB | Conservative / debugging |
| **2** | **16** | **131,072** | **~10.6 GB** | **Default (recommended)** |
| 4 | 8 | 131,072 | ~11.2 GB | Higher throughput |
| 8 | 4 | 131,072 | ~12.4 GB | Maximum throughput (still 67 GB headroom) |

---

## 8. Frozen Checkpointing Strategy

```yaml
checkpoint:
  format: "safetensors"
  precision: "bf16_weights_fp32_optim"
  contents:
    - "model_state_dict"           # bf16, ~1.74 GB
    - "optimizer_state_dict"       # fp32, ~4.99 GB (for active params)
    - "scheduler_state"
    - "step"
    - "token_count"
    - "best_loss"
  save_interval_steps: 2000
  save_async: false
  save_max_keep: 3
  resume_from: null               # explicit, not auto
  warm_start_weights: null
```

---

## 9. Frozen Data Pipeline

```yaml
data:
  packing: true                   # always pack to seq_len=4096
  pad_token_id: 0
  bos_token_id: 1
  eos_token_id: 2
  shuffle: true
  shuffle_buffer: 10000
  drop_last: true
  num_workers: 4
  prefetch_factor: 2
  mmap: true
  validation_split: 0.001         # 0.1% held out
```

---

## 10. Frozen Profiling and Observability

```yaml
profiling:
  enabled: true
  profile_steps: [100, 500, 1000]  # one-shot profiling at these steps
  profile_output: "profiles/pretrain"
  track:
    - "tokens_per_sec"
    - "step_time_ms"
    - "vram_peak_gb"
    - "grad_norm"
    - "loss_main"
    - "loss_mtp1"
    - "loss_mtp2"
    - "expert_load_entropy"
    - "expert_load_max"
    - "gdn_state_max_abs"
```

---

## 11. Frozen Risk Mitigations (cross-reference to CONVERGENCE_RISK_ASSESSMENT.md)

| Risk | Mitigation |
|------|------------|
| MoE routing collapse | Aux-loss-free biased sigmoid with `bias_update_every=10`; floor balance loss `alpha=1e-4` |
| GDN state overflow | FP32 state cast; chunk size capped at 64; A_log reinit on NaN |
| Attention logit explosion | QK-Norm enabled; logit softcap 15.0 on all heads |
| Gradient checkpointing numerical drift | BF16 forward + BF16 recompute (no upcast); spot-check vs no-checkpoint every 1000 steps |
| μP transfer failure | Verified base shapes (768, 24) match μP reference; LR scaled accordingly |
| Loss spike | Auto-rollback to last checkpoint, reduce LR by 2×, skip batch |
| OOM on long-tail batch | `empty_cache_every=100`; refuse to start if `min_vram_gb < 70` |

---

## 12. Implementation Phases (UNCHANGED from v1)

```
Phase 0: Repository cleanup
Phase 1: Dependencies (pyproject.toml, requirements.txt)
Phase 2: Models (transformer, mla, gated_deltanet, moe, mtp, mup)
Phase 3: Training loop (config, trainer, pretrain, train_step, optimization)
Phase 4: Utilities (device_setup, checkpoint/manager, logging)
Phase 5: Data (async_loader, prepare_data)
Phase 6: Config & scripts (configs/pretrain.yaml, scripts/run_pretrain.sh, README, docs)
Phase 7: Validation (smoke_test, A100 80GB fit check, param count verification)
```

---

## 13. Approval and Immutability

| Item | Status | Value |
|------|--------|-------|
| Architecture | **FROZEN** | This document |
| Active params | **FROZEN** | ~415.6M |
| Total params (all experts) | **FROZEN** | ~868.6M |
| VRAM (BS=2, T=4096) | **FROZEN** | ~10.6 GB |
| Chinchilla tokens | **FROZEN** | 8.31B |
| Wall-clock | **ESTIMATED** | 5–10 days (midpoint 5.2) |
| Training recipe | **FROZEN** | WSD + NorMuon + CautiousAdamW |
| Layer schedule | **FROZEN** | GDN every 3rd layer |
| MoE config | **FROZEN** | 8 routed + 1 shared, top-2, inter=2048 |
| GDN config | **FROZEN** | d_inner=1024, d_state=32, headdim=32 |
| MTP config | **FROZEN** | depth=2, weights [0.10, 0.05] |

**After this document**: Implementation begins. No further architectural changes permitted in v1.

---

*End of FINAL_FROZEN_SPEC.md*
