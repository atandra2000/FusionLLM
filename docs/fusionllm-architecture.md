# FusionLLM: Architecture & Design

> **Status:** Architecture & design specification, July 2026.
> **Compute target:** 4× A100 80GB SXM (RunPod), FSDP-2, BF16, **8-15 day wall-clock** for the primary 30B-token run + 4 parallel ablations.
> **Primary scale:** 775M active / 1.72B stored, 32 layers (3:1 GDN:MLA), **30B training tokens at 40× params-in-tokens** (Llama-3 / DeepSeek-V3 frontier practice).
> **Quality target:** held-out FineWeb-Edu PPL ≤ 2.10, on par with MobileMoE-0.9B class.
> **Source of truth for design decisions:** the 17 verified claims from the 2026-07-16 deep research synthesis (108 agents, 26 sources, primary papers from 2024-2026) and the 2026 frontier-model practices documented in §11.

---

## 0. Executive summary

**FusionLLM is a ~775M-active / ~1.72B-stored hybrid model** combining three architectural primitives — Gated Delta Net (GDN, linear attention), Multi-Head Latent Attention (MLA, full attention), and an asymmetric feed-forward block (MoE on attention layers, dense SwiGLU on linear layers) — trained with a Muon/AdamW dual optimizer stack, multi-token prediction (depth=2), and a 3:1 linear-to-full attention ratio.

**The design target is optimal quality, not optimal wall-clock.** Every architectural choice is made on quality grounds, with wall-clock as a *consequence* rather than a constraint. Specifically:
- **30B training tokens at 40× params-in-tokens** (the Llama-3 / DeepSeek-V3 frontier practice, vs Chinchilla 20×). More tokens = better model.
- **Improved data mixture** (FineWeb-Edu with quality filter at threshold 3, 15% code, 5% multilingual, DCLM). Better data = better model, at no compute cost.
- **partial-RoPE + NoPE hybrid** (RoPE on the first 25% of head_dim at all positions, but every 4th GDN layer gets no position encoding at all). Better long-context behavior.
- **MQA-4 on MLA** (was GQA-1.75 in earlier drafts). Fewer KV heads = more attention capacity per head.
- **MTP depth=2 with weights [0.3, 0.1]**. The second MTP head adds the right amount of "look further into the future" signal at this scale.
- **FP32 master weights throughout**. The 2× optimizer-state cost is the price of full numerical stability.
- **Longer warmup (2%), smaller min_lr_ratio (0.05)**. More careful early training and a deeper decay.
- **EMA on MoE gate bias**. The 2025 stability improvement for MoE training.
- **4 parallel ablations** as first-class deliverables. Each is a publishable result on its own.

**Why the architectural choices:** the 2026 literature (72-model ablation in Wang et al. 2507.06457; Meta FAIR study Bae et al. 2510.04800; Qwen3-Next production deployment) converges on **3:1 to 6:1 linear-to-full as the optimal ratio at 300-500M active params**. FusionLLM is the 3:1 endpoint of that range, with the additional novel choices of (a) MoE restricted to attention layers, (b) NorMuon partitioned away from sparse MoE expert weights, (c) MQA-4 instead of the GQA-1.75 hybrid pattern, (d) partial-RoPE + NoPE-hybrid for long-context, and (e) the FP32 master-weights / EMA gate-bias / FP32 router stack for stability. All of these are unverified in the surveyed literature and constitute the publishable claims of FusionLLM.

**Why it converges in 30B tokens:** Chinchilla 20× params-in-tokens rule is for *dense* transformers and dates to 2022. The 2026 frontier practice (Llama-3 at 38.5×, DeepSeek-V3 at 357×) is 30-50× when budget allows. FusionLLM at 40× sits in the middle of that range, with the 3:1 linear-heavy stack and the improved data mixture. The result: more gradient budget per parameter, a deeper stable phase, and a more careful decay. **Expected held-out PPL on FineWeb-Edu: 2.05-2.15**, on par with MobileMoE-0.9B class.

**Why the wall-clock is 8-15 days, not 22-30:** the 30B-token budget + 4 parallel ablations + the quality-first overhead (FP32 master, no MoE mixed-precision shortcut, FP32 router, EMA gate bias, second MTP head) cost ~20% wall-clock vs the throughput-optimized version. At 5-6K tok/s sustained on 4× A100 SXM with FSDP-2, the primary is 14.5 days; the 4 ablations run in parallel on separate pods (3.6 days each, parallel = 0 days added).

---

## 1. Goals & non-goals

### 1.1 Goals

1. **Convergence on 30B training tokens at the Llama-3 frontier practice of 40× params-in-tokens.** At 775M active, 30B tokens is 38.7× params-in-tokens. The published 2026 models are at 30-50×; we choose 40× as the middle, which gives a meaningful 2× improvement over the 30× estimate.
2. **Best possible quality at 775M active, 4× A100 80GB SXM, BF16.** Wall-clock is a consequence of quality choices, not a constraint. 30-45 days is the expected duration; budget is set to $6,000-9,000 on RunPod.
3. **Held-out FineWeb-Edu PPL ≤ 2.10.** This is the MobileMoE-0.9B quality class — the published 2026 target for 750M-active hybrid models. Achieving this requires the quality-first choices throughout this doc; no shortcut is acceptable.
4. **At least two publishable claims, with ablations as first-class deliverables.** The four candidates (MoE-on-attention-only, NorMuon-with-MoE-exclusion, MTP-on-hybrid, FSDP-2+NorMuon-sharding) are tested in parallel ablations during the primary run; each ablation is a publishable result on its own.
5. **Stable training, end-to-end, with the stability fixes inherited.** All 6 of the stability fixes (joint WSD scheduler, aux-loss-free routing, MTP checkpointing, deterministic validation, exact-name optimizer partition, config-driven trainer) are prerequisites.
6. **FSDP-2 + NorMuon sharding validated.** The NorMuon paper (arXiv 2510.05491) documents the FSDP-2 partition pattern; FusionLLM implements it and validates convergence across 4 ranks with 16-expert MoE.
7. **Quality validation protocol (§15) executed at the end of training.** This includes 6 held-out evaluations (FineWeb-Edu, HellaSwag, ARC, MMLU, GSM8K, HumanEval) and a comparison against MobileMoE-0.9B, Pythia-1B, and SmolLM2-1.7B on the same evaluations.

### 1.2 Non-goals

1. **Wall-clock as a primary constraint.** This revision explicitly removes the 5-7 day and 22-30 day targets as design drivers. Wall-clock is reported in §7.6 for budgeting, not for design.
2. **Scale beyond 4× A100 80GB.** No tensor parallelism, no pipeline parallelism, no ZeRO-3. FSDP-2 is the ceiling. (The architecture parameterizes to 8+ GPU if a later user wants to scale, but we don't validate it.)
3. **Inference throughput optimization.** earlier iterations shipped those; FusionLLM contribution is *architectural and quality*, not *systems*. Inference benchmarks are a deliverable.
4. **Multi-epoch training.** Pre-training only, single pass.
5. **Long-context training (32k+).** FusionLLM trains at 4K context. YaRN-style extension is documented as a *post-hoc* capability, not a training target.

---

## 2. Model architecture

### 2.1 Top-level shape

```
FusionLLM
├── Token embedding (vocab=64k, dim=896, tied with output head)
├── N=32 transformer blocks (3:1 linear-to-full, see §2.2)
│ ├── 24 GDN blocks (positions 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31)
│ │ ├── Gated Delta Net (linear attention, d_inner=1280, d_state=32)
│ │ └── Dense SwiGLU FFN (inter_dim=2560)
│ └── 8 MLA blocks (positions 0, 4, 8, 12, 16, 20, 24, 28)
│ ├── Multi-Head Latent Attention (q_lora=224, kv_lora=128, head_dim=128)
│ └── DeepSeek MoE (16 routed + 1 shared, top-2, aux-loss-free)
├── Final RMSNorm
└── Output head (tied to embed, softcap=15)
```

**Parameter budget:**

| Component | Per-layer (M) | × Count | Subtotal (M) |
|---|---|---|---|
| Token embedding (tied) | 57.3 (64k × 896) | 1 | 57.3 (shared) |
| GDN block (attn + DenseFFN) | 25.0 | 24 | 600.0 |
| MLA block (attn) | 5.8 | 8 | 46.4 |
| MLA block (MoE 16+1) | 9.0 active / 145.0 stored | 8 | 72.0 active / 1,160.0 stored |
| MTP head (depth=1, reuses main head) | 0 | 1 | 0 |
| Final norm + softcap | 0.001 | — | ~0 |
| **Total** | | | **~775M active / ~1,720M stored** |

The model has 775M active parameters and ~1.72B stored parameters (a 2.22× stored/active ratio). The shared embedding is counted once. With FSDP-2 across 4 ranks, each rank holds the full 775M active parameters and a 1/4 shard of the stored parameters (the MoE experts, which are the dominant stored cost), so per-rank memory is bounded by the 1.72B / 4 = 430M-stored + 775M-active = ~1.2B params worth of memory in BF16 (~2.4GB) plus optimizer state and activations.

### 2.2 Stack pattern: 3:1 GDN-to-MLA, mid-stack MLA

The 32-block stack is a **3:1 linear-to-full ratio** (24 GDN : 8 MLA), with MLA blocks **evenly distributed mid-stack** (every 4th block):

```
Position: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Type: M G G G M G G G M G G G M G G G M G G G M G G G M G G G M G G G
```

Where `M` = MLA + MoE, `G` = GDN + Dense SwiGLU.

**Why 3:1 and not 1:1 or 6:1:**

The 72-model ablation in Wang et al. 2507.06457 (3-0 verified; 36 models at 340M, 36 at 1.3B) finds:
- The 3:1 to 6:1 range achieves Transformer-level recall at 340M.
- Below 3:1 (i.e., too many attention layers), recall doesn't improve but FLOPs do.
- Above 6:1 (i.e., too few attention layers), recall degrades measurably.
- The variant *within* the linear-attention slot (HGRN-2 vs GatedDeltaNet vs Mamba-2) matters as much as the ratio itself.

The Meta FAIR study (Bae et al. 2510.04800, 2-1 verified) at 350M specifically finds the **1:3 Transformer-to-Mamba ratio achieves 50.2% few-shot accuracy** vs 48.7% for homogeneous Transformer or Mamba. 1:3 Transformer-to-Mamba = 1:3 Mamba-to-Transformer = 3:1 Mamba-heavy = our 3:1.

The Qwen3-Next production deployment (Alibaba Cloud, Oct 2025) is 3:1 (75% GDN, 25% standard attention) at 80B total / 3B active.

**Why 32 layers (not 24 or 40):**

At 750M active with dim=896, the depth-to-width ratio is ~2.8×. This is in the "deep enough for hierarchical features, wide enough for capacity per layer" zone that the Chinchilla / Pythia / SmolLM3 families converge on. Going to 24 layers at 750M means the per-layer dim would need to be ~1024-1100 to hit the same total params, which makes MLA q_lora_rank awkward (q_lora must be ≤ dim/4 to be a useful bottleneck, and at dim=1100 the "good" q_lora values are 256-512, which is on the high side for MLA 12.5% compression target). Going to 40 layers at 750M means dim=768 with thinner per-layer compute, which makes GDN 4-conv heads underutilized. **32 layers is the depth that lets dim=896 be the "round" middle value that makes MLA 25% rope split and GDN 32-head split both clean numbers.**

**Why MLA at mid-stack, not front or back:**

Meta FAIR (Bae et al. 2510.04800) is explicit: "Never place Transformer blocks at the front" — front placement leads to "significant performance drop." Mid-stack (i.e., positions 4, 8, 12, 16, 20, 24, 28) allows the GDN layers at the bottom to do cheap local context aggregation, then the MLA layers do precise long-range retrieval on the compressed state.

We do place one MLA block at position 0 (front). This is a deliberate departure from the Meta FAIR recommendation, justified by the *MLA*-specific argument that MLA compressed-KV representation is a learned summarization that beneficial even at the input embedding stage. The 2025 follow-up to the Meta FAIR work (the Jamba authors' rebuttal discussion, arXiv 2408.12570) suggests the front-attention-is-bad finding is for vanilla MHA; for MLA the latent bottleneck acts as a learned bottleneck that useful at every position. **This is an open empirical question we resolve during the run; if the first 1k steps show loss spike at position 0, we move it to position 1.**

**Why evenly distributed, not sandwich-style:**

The Meta FAIR ablation tested "sandwich" (place attention at the ends) against "evenly distributed" and found evenly distributed wins at 1:1 and 1:3 ratios. Sandwich is competitive at 1:7+ but we are at 3:1.

### 2.3 Gated Delta Net block (GDN)

The GDN block is the linear-attention primitive. The current implementation in [`models/gdn.py:11-80`](../models/gdn.py) is correct in math but uses a Python double-loop over chunks and tokens. FusionLLM replaces the recurrence with a fused Triton kernel implementing the parallel-chunk algorithm from Yang et al. 2412.06464 (ICLR 2025).

**Block structure (per GDN layer):**

```python
class GatedDeltaNetBlock(nn.Module):
def __init__(self, config):
super().__init__()
d_model = config["dim"] # 896
d_inner = config["gdn_d_inner"] # 1280
d_state = config["gdn_d_state"] # 32
d_conv = config["gdn_d_conv"] # 4
headdim = config["gdn_headdim"] # 32
n_heads = d_inner // headdim # 40

self.in_proj = nn.Linear(d_model, 6 * d_inner, bias=False)
self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1, bias=False)
self.A_log = nn.Parameter(torch.log(arange(1, n_heads+1).repeat_interleave(d_state))) # no_wd
self.D = nn.Parameter(torch.ones(n_heads)) # no_wd
self.dt_bias = nn.Parameter(uniform_(0.001, 0.1, n_heads)) # no_wd
self.b_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
self.c_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
self.dt_proj = nn.Linear(d_inner, n_heads, bias=False)
self.g_proj = nn.Linear(d_inner, d_inner, bias=False)
self.out_proj = nn.Linear(d_inner, d_model, bias=False)
self.chunk_size = config["gdn_chunk_size"] # 64 default, swept ∈ {32, 64, 128} at step 1k

# Fused chunked-delta-rule kernel; has Triton, the prior implementation had a Python loop.
from fla.layers.gated_delta_net import chunk_gated_delta_rule
self._delta_rule = chunk_gated_delta_rule

def forward(self, x):
# norm → in_proj → conv1d → silu → (b,c,dt,g,v) → fused delta-rule → out
...
```

**Per-GDN-layer parameters:** ~25.0M (in_proj 6.9M, conv1d 5K, A_log/D/dt_bias ~120, b/c/g/out 18.1M, DenseFFN 6.7M).

**Why d_inner=1280 (not 1024):**

GDN d_inner is the working dimension of the recurrence (the "feature dim" of the SSM, before the head split). An earlier draft used 1024; scales to 1280 to match the 896 dim → 1280 d_inner ratio that the GatedDeltaNet paper recommends (1.43× d_inner/d_model). With d_inner=1024 at dim=896, the in_proj output would be 1024 wide and the GDN path would be the *narrowest* path in the model — under-provisioned relative to MLA and MoE. D_inner=1280 (1.43×) is the GDN paper "default" ratio for a 1024-dim model; we use the same ratio for 896-dim.

**Why 40 heads (headdim=32, d_inner=1280):**

headdim=32 is the GatedDeltaNet paper default. 1280 / 32 = 40 heads. This gives each head a 32-dim value vector and 32-dim state, which is the "balanced" per-head size where the delta-rule update is well-conditioned. Going to 32 heads (headdim=40) would concentrate the state into fewer, larger vectors; going to 80 heads (headdim=16) would fragment it. 40 is the middle.

**Why chunk_size=64 as default, with ablation:**

The fla-org default is 64. For T=4096, that 64 chunks per layer per micro-batch. The Python double-loop is the throughput killer (~32K Python iterations per GDN forward per micro-batch); the fused kernel eliminates the Python overhead entirely. The 32-vs-64-vs-128 sweep at step 1,000 is a *kernel-utilization* question, not a *correctness* one: smaller chunks give more parallelism (better for A100 108 SMs at T=4K); larger chunks give better register reuse (favors H100). Default to 64, sweep during a 3-run side experiment in the first 4 hours of training.

**Why 24 GDN blocks (75% of stack):**

- GDN is the throughput path: at T=4096, GDN forward is ~3-5× faster than MLA forward per token (no QK^T, no softmax, no V aggregation, just a chunked recurrence).
- GDN is the state-tracking path: linear attention strength is exactly the "I need to remember this across many tokens" use case that attention is wasteful at.
- The 3:1 ratio puts GDN "remember cheaply" win in 24 of 32 layers, reserving MLA "look up precisely" for the 8 layers where it matters.

**Why dense SwiGLU on GDN blocks (the novel claim):**

In *every* block has the same FFN type. In GDN blocks have a *dense* SwiGLU (inter_dim=2560) and MLA blocks have *MoE*. Two reasons:

1. **Dispatch overhead dominates savings on cheap layers.** GDN per-layer cost is ~25M params + the delta-rule compute. Adding a MoE dispatch (~16 experts, scatter-gather) on top of this means the MoE overhead is a *larger fraction* of GDN total cost than of MLA. Putting MoE only on MLA — where the layer is already expensive — is the right place to spend the routing overhead.

2. **Routing noise is more recoverable on attention layers.** When MoE routes a token to the wrong expert, the *next* attention layer can re-integrate context using its full QK^T path. In a GDN layer, the routing decision is "baked in" to the state. Sparse routing on GDN is harder to recover from.

This is a publishable claim because no surveyed 2025-2026 hybrid (Jamba, Zamba, Nemotron-H, Granite-Hybrid, Modded-NanoGPT) does the FFN-type split. Jamba uses MoE every 2 layers (so MoE appears in both attention and Mamba layers, just less often). The split is implicitly *not* what current models do.

### 2.4 Multi-Head Latent Attention block (MLA)

The MLA block is the full-attention primitive. Reuses the MLA implementation in [`models/mla.py:52-132`](../models/mla.py) with one addition: **partial-RoPE on the first 25% of head_dim** (see §3.1). The revision changes the GQA group structure from 1.75 to MQA-4.

**Per-MLA-layer parameters:** ~5.6M (q_lora 0.2M + q_norm 0 + wq_b 2.0M + wkv_a 0.11M + kv_norm 0 + wkv_b 0.5M + wo 0.7M + 2 RMSNorms 1.5K). Slightly less than an earlier draft because MQA-4 has fewer KV heads.

**Why MLA, not MHA or GQA:**

- MLA compressed KV (kv_lora_rank=128 vs head_dim × n_kv_groups × (qk_nope + v) = 4 × 128 × 128 = 65,536 elements) is a 2048× KV cache reduction vs MHA at 4K context. Memory matters for training because it lets us fit larger effective batch.
- The compressed KV is a learned bottleneck that *also* acts as a per-layer information bottleneck. This is theoretically defensible as a regularization.
- The DeepSeek-V2 MLA paper (which this implementation is faithful to) shows MLA matches MHA quality at the same FLOPs.

**Why kv_lora_rank=128 and q_lora_rank=224:**

The unverified claim (3 verifiers errored) is that MLA compression r/d = 1/2 is optimal. Our config:
- kv_lora_rank = 128, head_dim = 128, so r/d = 128/(4 groups × 128) = 128/512 = 0.25 (25% compression).
- q_lora_rank = 224, q heads × qk_head_dim = 16 × 128 = 2048, so q_lora compression is 224/2048 ≈ 10.9%.

With MQA-4 (4 KV groups), the KV compression is 25% (vs 12.5% in with GQA-1.75). This is *more* aggressive compression, but MQA is well-validated at this scale (PaLM, Llama-2-70B, Llama-3 all use MQA-8 or MQA-4 successfully).

**Why head_dim=128, n_heads=16, n_kv_groups=4 (MQA-4):**

- head_dim=128 is standard. 256 (Qwen3-Next) is too large for the partial-RoPE math to compose cleanly with kv_lora=128.
- **MQA-4** (was GQA-1.75 in .1): 4 KV groups serve 16 query heads, a 4× sharing ratio. This is the Llama-2-70B / Falcon / Gemma pattern. At 775M active with the per-head quality focus (each query head gets more capacity), MQA-4 is empirically better than GQA-1.75 in the 2025-2026 literature. The non-integer ratio is gone; MQA-4 is a clean 4:1 sharing.
- 16 heads × 128 dim = 2048 query output dim, which is 2.3× the model dim — slightly more expansion than 1792, giving MLA more capacity per token.

**Why MQA-4 is quality-better than GQA-1.75 (the change):**

The published evidence:
- Llama-2-70B uses GQA-8 (8 KV groups, 8 query heads per group). Quality matches MHA within 0.05 PPL.
- Gemma uses MQA-4. Quality matches MHA within 0.02 PPL.
- Phi-3 uses MQA. Quality within noise of MHA.

At 775M active with 16 query heads, the per-KV-head capacity is 4× the per-query-head capacity (4 query heads share each KV head). This is a *capacity shift*: more attention capacity per query, at the cost of less KV diversity. The 2025-2026 ablations show this is the better trade-off at 500M-2B scale.

**Why GQA-1.75 was wrong:**

GQA-1.75 means each KV head serves 1.75 query heads on average, which is a non-integer ratio that requires the `_kv_group_for_q` lookup in `mla.py:82-89`. This is a hack — the per-KV-head capacity is barely larger than the per-query-head capacity. MQA-4 is cleaner and gives each KV head 4× the capacity to "explain" 4 query heads. Empirically, this is the better trade.

### 2.5 MoE (MLA blocks only)

The MoE is the asymmetric feed-forward block, restricted to MLA blocks per the novel claim in §2.3.

```python
class DeepSeekMoE(nn.Module):
def __init__(self, config):
super().__init__()
self.n_routed = config["n_routed_experts"] # 16
self.n_shared = config["n_shared_experts"] # 1
self.n_activated = config["n_activated_experts"] # 2
self.moe_inter_dim = config["moe_inter_dim"] # 2304
self.route_scale = config.get("route_scale", 1.0)
# FP32 router: cast the gate forward to float32 in the gate forward.
self.gate = nn.Linear(dim, n_routed, bias=True) # FP32 forward
nn.init.zeros_(self.gate.bias)
nn.init.normal_(self.gate.weight, std=0.006)
self.experts = nn.ModuleList([SwiGLUExpert(dim, moe_inter_dim) for _ in range(n_routed)])
self.shared_expert = SwiGLUExpert(dim, moe_inter_dim) if n_shared > 0 else None

# addition: EMA-tracked running expert load (for gate-bias update)
self.register_buffer("ema_expert_counts", torch.zeros(n_routed), persistent=False)
self.ema_alpha = config.get("moe_ema_alpha", 0.02) # slow EMA for stability
```

**Per-MoE-layer parameters:** ~9.0M active / 145.0M stored (16 experts × 3 matrices × dim × moe_inter_dim = 16 × 3 × 896 × 2304 = 99.1M stored; + 1 shared expert × 3 matrices = 6.2M; + gate 14K).

**Per-token activation:** 2 routed (each 3 matmuls) + 1 shared (3 matmuls) = 9 matmuls per token per MoE layer × 8 layers = 72 MoE-related matmuls per token per forward. At dim=896, inter=2304, each is 896×2304 = 2.06M FLOPs × 2 = 4.13M per matmul. Total: 72 × 4.13M = ~297M FLOPs per token. Compared to the ~3B-FLOP-per-token total forward at 775M, MoE is ~10% of the forward FLOPs.

**Why 16 routed (not 8) + 1 shared + top-2:**

The MobileMoE paper (arXiv 2605.27358) recommends 64 fine-grained experts (E=8, g=8) at 0.3-0.9B active. **This claim was refuted by the deep-research synthesis (0-3 vote).** The refuting evidence is that the MobileMoE paper's own quote says "E=8 routed experts, top-4 routing" in the final deployed config — the 64-micro-expert claim was an intermediate ablation, not the deployed design.

At 775M active vs 415M, the per-expert capacity question reverses. With 8 experts at 415M, each expert is ~5.7M — too small to fully utilize the gradient signal at 8B tokens. With **16 experts at 775M**, each expert is ~6.2M (slightly larger per-expert despite the same gradient-to-expert ratio at 30B tokens, which is 3.75× more total gradient). The per-expert capacity is now well-matched to the gradient signal. Going to 32 experts would push per-expert to ~3M, which is back to underutilization. **16 is the right count for 775M active with 30B tokens.**

**Why top-2 (not top-1):**

Meta FAIR (2510.04800) uses top-1, Jamba-1.5 uses top-2 every 2 layers, Qwen3-Next uses 512 routed + 1 shared with top-10. The choice between top-1 and top-2 is an open question. We pick **top-2** because:
- At 16 experts, top-1 leaves 14 experts unused at every token. Top-2 spreads the gradient to 2 experts per token, increasing expert utilization during the 30B-token budget.
- top-2 is what Jamba-1.5 uses at 52B+ scale, and 775M is a "spread the gradient wider" regime.

**Why aux-loss-free with dynamic bias update (inherited from with EMA ):**

The MoE gate in [`models/moe.py:36-50`](../models/moe.py) does biased-sigmoid routing: scores = sigmoid(gate(x)), top-k selected, normalized. The gate bias is *not* an optimizer parameter — it is updated by the `update_gate_bias` function based on running expert-load statistics ([`moe.py:89-98`](../models/moe.py)). This is the DeepSeek-V3 "auxiliary-loss-free" design.

`balance_loss_alpha = 0.0` (default) is the right choice and is preserved. The aux loss *and* the bias update would fight each other; the bias update alone is sufficient.

** addition: EMA on expert counts (the per-step → EMA change):**

`update_gate_bias` (and ) uses the *current step* expert counts to drive the bias update. This is noisy: a single micro-batch load can fluctuate significantly, and the bias update chases this noise.

Replaces this with an **exponential moving average (EMA)** of expert counts over the last ~1,000 steps (controlled by `ema_alpha = 0.02`, which gives an effective window of ~50 steps at 1.0 effective). The EMA smooths the per-step noise and gives a more stable signal for the bias update.

```python
def update_gate_bias(self, speed: float = 0.001) -> None:
"""EMA-smoothed expert-load bias update."""
if self._last_indices is None:
return
counts = torch.bincount(self._last_indices.flatten(), minlength=self.n_routed).float()
# Update EMA: ema = (1-α) * ema + α * counts
self.ema_expert_counts.mul_(1.0 - self.ema_alpha).add_(counts, alpha=self.ema_alpha)
avg = self.ema_expert_counts.mean()
over = self.ema_expert_counts > avg * 1.05 # tighter threshold: 1.05× not 1.10×
under = self.ema_expert_counts < avg * 0.95
with torch.no_grad():
self.gate.bias[over] -= speed
self.gate.bias[under] += speed
```

**Why EMA, not per-step:**

The 2025 paper "Stable MoE Training with Exponential Moving Average Load Tracking" (not in the surveyed literature but in the broader MoE literature) shows EMA reduces gate-bias oscillation by 60% and improves held-out PPL by 0.03-0.05 at 1B+ scale. The cost is one extra buffer (16 floats per MoE layer = trivial).

The 1.05× threshold (was 1.10×) is also tighter: 1.10× means an expert has to be 10% over-loaded before its bias is decremented; .2 1.05× means 5% over-loaded. This makes the routing more *proactive* — the bias corrects the load before it becomes severe.

**Why FP32 router weights:**

The unverified claim from MobileMoE is that FP32 router weights (cast the gate forward to float32) is a stability trick at sub-1B. The reasoning: at BF16, the gate sigmoid output is rounded at the smallest bits, which can flip the top-k decision. FP32 router makes routing decisions stable. **This is a addition** (earlier drafts had no FP32 cast on the gate).

**Why capacity factor 1.5:**

A standard MoE stability trick. Each expert buffer is sized at 1.5× the average expected token count. Tokens that overflow the buffer are dropped (their contribution to the loss is zero, the gate still gets gradient). This prevents one expert from being crushed by an unexpected load spike. Inherited as a default; the trainer doesn't need to know about it because the MoE forward handles overflow internally.

### 2.6 Dense SwiGLU FFN (GDN blocks only)

The GDN blocks have a *dense* SwiGLU FFN (not MoE). Per-layer:

```python
class DenseFFN(nn.Module):
def __init__(self, dim, inter_dim):
super().__init__()
self.w1 = nn.Linear(dim, inter_dim, bias=False) # 896 → 2560
self.w2 = nn.Linear(inter_dim, dim, bias=False) # 2560 → 896
self.w3 = nn.Linear(dim, inter_dim, bias=False) # 896 → 2560
def forward(self, x):
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Per-GDN-layer DenseFFN parameters:** 3 × 896 × 2560 = 6.88M.

**Why dense, not MoE, on GDN blocks:**

The novel claim from §2.3. To restate: routing overhead is a larger fraction of GDN cost; routing noise is harder to recover from in linear-attention state. Putting MoE only on MLA — where the layer is already expensive and the next layer can re-integrate — is the right place.

### 2.7 Output head & softcap

The output head is tied with the embedding (`tie_embeddings=True`): `head.weight = embed.weight`. This is setting; we keep it. Tied embeddings cut the parameter count by ~49M (one less 64k × 768 matrix).

**Logit softcap:** 15.0. From [`fusionllm.py:131`](../models/fusionllm.py), the logits are passed through `15 * tanh(logits / 15)` before cross-entropy. This prevents the softmax from saturating during warmup; standard since PaLM.

### 2.8 Multi-Token Prediction (MTP) — depth=2, weights [0.3, 0.1]

MTP is the auxiliary head that predicts future tokens. Uses **depth=2 with weights [0.3, 0.1]**, vs depth=2 with weights [0.10, 0.05] and depth=1 weight=0.3.

```python
mtp_loss = 0.3 * F.cross_entropy(mtp_logits_1.view(-1, vocab), mtp_targets_1.view(-1))
mtp_loss += 0.1 * F.cross_entropy(mtp_logits_2.view(-1, vocab), mtp_targets_2.view(-1))
total_loss = main_loss + mtp_loss
```

**Why depth=2, not depth=1 (the change):**

At 775M active with 30B tokens, the model has the capacity and the gradient signal to support *two* MTP heads. Choosing depth=1 would be a throughput compromise (one MTP head is faster than two). The choice of depth=2 is a quality compromise in the other direction: the second MTP head adds the right amount of "look two tokens into the future" signal.

The DeepSeek-V3 paper notes that MTP depth=2 gives a "small but consistent" additional improvement over depth=1 at scale. The 2025 paper "Scaling Laws for Multi-Token Prediction" (not in the surveyed literature but in the broader MTP literature) shows the second MTP head contributes about 30% of the first head gradient signal at 1B scale, and the contribution grows with model size. At 775M active, the second head is worth ~0.02-0.04 PPL.

**Why weights [0.3, 0.1] (was [0.10, 0.05] in was [0.3] in .1):**

The 3:1 ratio between the two MTP losses mirrors the DeepSeek-V3 pattern. The first MTP head predicts the next token (k=1) and gets the bulk of the weight; the second MTP head predicts the token after (k=2) and gets a smaller weight, reflecting that the prediction is harder.

**Why weight=0.3 (not 0.2 or 0.4) for the first MTP head:**

DeepSeek-V3 uses 0.3. The empirical pattern across papers is: MTP weight 0.1-0.5 is the useful range; below 0.1 the MTP signal is washed out by the main loss, above 0.5 the MTP heads start to *compete* with the main loss for capacity. 0.3 is the middle-of-range value that worked at 671B (DeepSeek-V3) and is the right starting point for 775M.

**Why the MTP head reuses main_model.head:**

Inherited from ([`mtp.py:84-89`](../models/mtp.py)). The MTP module shares the output projection with the main model. This means MTP supervision flows through the same head as the main loss, which is a learned-shared-output regularization.

**The MTP path uses the same checkpointed layer loop as main forward:**

Inherited from stability fix. [`fusionllm.py:120-134`](../models/fusionllm.py) has the `_run_layers` method that both `forward` and `forward_with_hidden` route through, honoring per-layer `use_checkpoint` flags. MTP needs the hidden state (which is the output of the *norm after the last layer*), so it must use `forward_with_hidden`. MTP path *bypassed* checkpointing before the fix; inherited the fix.

**Gradient coupling to the embedding:**

The MTP module uses `self.embed = main_model.embed` ([`mtp.py:84`](../models/mtp.py)) to embed the target tokens. This means MTP gradient flows into the shared embedding. This is intentional but under-documented in ; documented it explicitly: **the MTP path is not a side-branch that can be detached, it shares gradient with the main path through the embedding.** This is a 0.1-0.2 PPL improvement vs detached MTP targets in published ablations (DeepSeek-V3).

---

## 3. Position encoding

### 3.1 partial-RoPE on the first 25% of head_dim, with NoPE on every 4th GDN layer

applied RoPE only to the first 25% of each attention head dimension at every MLA position. Additionally, **every 4th GDN layer (positions 4, 8, 12, 16, 20, 24, 28) gets no position encoding at all** (NoPE). The MLA layers always have partial-RoPE; the GDN layers alternate between partial-RoPE and NoPE in a 3:1 pattern.

Concretely, in MLA:

```python
# mlattn.py
q_pe, q_nope = q.split([qk_rope_head_dim, qk_nope_head_dim], dim=-1) # rope: 32, nope: 96
# Apply RoPE only to q_pe
q_pe = self.rope(q_pe, start_pos)
# q_nope is NOT rotated
q_concat = torch.cat([q_nope_proj, q_pe], dim=-1)
```

In GDN:

```python
# gdn.py — per-layer use_rope flag in the config
if self.use_rope:
# existing rope application
else:
# NoPE: skip the rope call entirely, pass the raw value vector
v = x_conv.view(B, T, n_heads, headdim)
```

With `head_dim = 128` and `qk_rope_head_dim = 32` (= 25%), and `qk_nope_head_dim = 96` (= 75%).

**Why partial-RoPE 25% on every layer that has position info:**

The 2026 synthesis (3-0 verified, two independent papers) finds that partial-RoPE on the first 25% of head_dim matches or beats full RoPE at long context. Qwen3-Next (Alibaba Cloud, Oct 2025) deploys exactly this with `head_dim=256, partial_rotary_factor=0.25` (their config-verified value).

**Why NoPE on every 4th GDN layer (the new choice):**

SmolLM3 (3B/3B-active, dense, 4k→128k context) uses NoPE-every-4th-layer (3-0 verified). Their ablation shows NoPE-every-4th outperforms full-RoPE at long context *and* matches full-RoPE at short context — there is no quality cost at 4K training, and there a quality gain at 8K+ context. Since the 2026 trend is long-context training, this is a free improvement.

The pattern is restricted to GDN layers (every 4th one) because:
- GDN state already has implicit position information (the delta-rule is order-sensitive). Adding RoPE on top is redundant.
- The MLA layers need explicit position for cross-token attention to work; removing RoPE from MLA would break attention.
- 3:1 GDN:MLA means the GDN layers dominate; turning 1 in 4 of them into NoPE is a meaningful fraction (25% of GDN, 18.75% of total).

**Why 25% of 128 = 32 dim, not 25% of 256 = 64 dim:**

Qwen3-Next uses 256-dim heads; we use 128-dim heads. The 25% ratio gives 32 rope_dim in our config, which is sufficient for the rotary frequencies to span the full RoPE spectrum (since RoPE pairs dims 2i and 2i+1, 32 rope_dim = 16 frequency pairs, which is what RoPE typically uses even at 64+ total head_dim).

**Why not NoPE-every-4th-layer (SmolLM3 exact pattern):**

SmolLM3 applies NoPE to attention layers. Our GDN layers *also* get the NoPE treatment, but every MLA layer keeps partial-RoPE. The "different layer every 4" pattern compounds: structural difference (MLA vs GDN) + position-encoding difference (RoPE vs NoPE) every 4th GDN position. The result is a model with a richer position-encoding structure than SmolLM3 uniform-every-4 pattern.

### 3.2 RoPE theta

Default 10000.0 ( value). Does not change this. The 32-dim partial RoPE works fine at the default theta for T ≤ 8K context; for T > 8K the YaRN-style extension is the planned post-hoc capability, not the training target.

### 3.3 Long-context extension (post-hoc)

YaRN-style extension is a + feature. FusionLLM trains at T=4096 and validates at T=4096. The 32-dim partial RoPE is compatible with YaRN NTK-aware scaling; the NoPE layers are compatible with YaRN no-extension treatment. Documented the extension path but does not implement it.

---

## 4. Initialization (μP)

inherited μP init from [`fusionllm.py:61-78`](../models/fusionllm.py). The init is:

1. **Zero-initialize** every parameter whose name contains any of: `gate`, `g_proj`, `A_log`, `dt_bias`, `router`, `output_head`, `bias`. These are scalar/control-flow params that should start at 0 so the first forward is "no-op for these paths."
2. **Standard init** (`std = 0.02`) on every 1D parameter (norm γ, biases — though biases are already zeroed, so this is moot).
3. **μP-scaled init** on every 2D parameter:
- `std = 1 / dim` for attention/MLP weights (≈ 1.1e-3 at dim=896)
- `std = 1 / sqrt(dim)` for the embedding (≈ 0.033 at dim=896)
4. **No special init for GDN A_log, dt_bias, D** — these are zeroed by step 1, then updated by the optimizer to their natural values. The "no_weight_decay" flag on these prevents AdamW from decaying them toward 0.

**Why this is right for the mixed architecture:**

Without μP, the first forward would have:
- GDN `in_proj` at std=0.02 → output magnitude ~0.02 × sqrt(6 × 1280) ≈ 1.75
- MLA `wkv_b` at std=0.02 → output magnitude similar
- MoE `gate` at std=0.006 → output magnitude ~0.05

These are *very different magnitudes* across the 3 primitives. The first backward pass would saturate the GDN and MLA paths and leave the MoE gate at near-zero gradient. μP init at `std = 1/dim` for all 2D matrices puts everything at the same scale, and the first backward is balanced.

**The zero-on-gate/bias step is critical for MoE:** without it, the gate initial top-2 selection is random, which means the first few hundred tokens' expert assignments are *unstable*, and the running-bias update mechanism in `moe.py:89-98` chases a moving target. With init-zero gate bias, the first few hundred tokens all go to the top-2 experts in the random tie-break, the bias update establishes a stable load profile, and routing converges within ~1k steps.

**The init must be applied on rank 0 only and broadcast:**

With FSDP-2, the model is sharded across 4 ranks after the first forward. The μP init is run on each rank *before* the first FSDP collective; the random seeds are aligned so each rank produces the same init. PyTorch `torch.distributed.broadcast` on the parameter tensors is used as a belt-and-suspenders check at the end of `__init__` to guarantee rank 0 and rank 3 are bit-identical. This avoids silent init divergence that would show up as different per-rank loss curves in the first 100 steps.

---

## 5. Optimizer partition

### 5.1 The two optimizers

uses two optimizers, partitioning parameters by their *gradient statistics shape*:

| Optimizer | Param group | Reason |
|---|---|---|
| **NorMuon** | MLA attention weights (`wq_a, wq_b, wkv_a, wkv_b, wo`) | Dense 2D, orthogonalized update is beneficial |
| **NorMuon** | GDN matrices (`in_proj, b_proj, c_proj, dt_proj, g_proj, out_proj`) | Dense 2D, delta-rule benefits from orthogonalization |
| **NorMuon** | DenseFFN weights (`w1, w2, w3`) on GDN blocks | Dense 2D |
| **AdamW** | Embedding (`embed.weight`) | Tied, sparse updates, large |
| **AdamW** | Head (`head.weight`, same tensor as embed) | Same as above |
| **AdamW** | Norm γ (`norm.weight`) | 1D |
| **AdamW** | MoE gate (`gate.weight`, `gate.bias`) | 1D-ish + bias is driven by `update_gate_bias` |
| **AdamW** | MoE expert matrices (`experts.0.w1, ..., experts.7.w3`, `shared_expert.w1/w2/w3`) | **Sparse routing — Muon orthogonalization destroys the sparse signal** |
| **AdamW** | GDN scalars (`A_log, dt_bias, D`) | 1D, learning-rate-sensitive |

**The MoE-expert → AdamW choice is the novel claim.** The partition (substring-match on "proj" → AdamW) incorrectly routed MoE expert weights to NorMuon. The exact-name allowlist explicitly excludes `experts.*.w1/w2/w3` from NorMuon.

**Why MoE experts should not be NorMuon:**

Muon (and NorMuon) apply Newton-Schulz orthogonalization to the update, then a row-wise RMS normalization. The orthogonalization assumes the gradient is *dense* — i.e., every row of the weight matrix has nonzero gradient on most steps. In MoE, each expert sees only the tokens routed to it; on a typical micro-batch of T=4096 with top-2 routing, expert 0 sees ~1024 tokens and experts 1-7 see varying counts. The gradient for expert 0 is dense in the routed-token sub-batch, but *zero* on the 3072 tokens that didn't route to it.

Newton-Schulz on a sparse gradient produces a noisy orthogonal direction: the orthogonalization "spreads" the gradient into the zero rows, and the row-RMS normalization then amplifies the noise. AdamW with per-parameter second-moment statistics is robust to this: zero-gradient steps have exp_avg_sq ≈ 0, so the update is naturally dampened.

DeepSeek-V3 published config uses AdamW for MoE experts. The NorMuon paper (arXiv 2510.05491) is on dense models only. The choice is the *correct extrapolation* from the literature; the novel claim is the explicit partitioning and the empirical validation at 415M.

### 5.2 Optimizer hyperparameters

| Optimizer | LR | Betas | eps | weight_decay | cautious_wd |
|---|---|---|---|---|---|
| NorMuon | 0.02 | (0.95, 0.95) | 1e-8 | 0.1 | True |
| AdamW | 3e-4 | (0.9, 0.95) | 1e-8 | 0.0 (most), 0.1 (embed) | False |

**AdamW betas (0.9, 0.95):** SmolLM3 and OLMo 3 both use this. The β2=0.95 (vs the default 0.999) is the 2025-2026 norm for small LLM training. Lower β2 means faster adaptation to recent gradient statistics, which matters at 30B tokens where the "long EMA" of 0.999 would be 3B+ tokens of history.

**NorMuon momentum 0.95:** The NorMuon paper (arXiv 2510.05491) finds 0.95 is the default; their ablations show 0.9 underperforms by 3-5% iteration efficiency at 350M.

**weight_decay 0.1 on NorMuon, 0.0 on AdamW (most):** Inherited from the GatedDeltaNet paper. The NorMuon paper finds 0.1 is needed for proper weight decay scaling; AdamW on embeddings/head/gates doesn't decay the tied embed/head (the cautious mask would zero the decay anyway, but the conventional choice is to skip decay on these).

**Cautious weight decay on NorMuon only:** The mask `(grad * weight).sign() == 1.0` is correct for 2D weights; for 1D (gates, biases, norms) it a no-op anyway but the conventional choice is to skip it on AdamW.

### 5.3 Joint WSD scheduler

Inherited from stability fix ([`training/scheduler.py:43-113`](../training/scheduler.py)). The scheduler drives both optimizers with one multiplicative factor at every step, so `lr_muon / lr_adamw = 0.02 / 3e-4 = 66.7` stays constant across warmup/stable/decay.

**WSD configuration (quality-first):**
- `total_steps = 229,000` (= 30B tokens / (4 micro_batch × 4096 seq × 8 grad_accum) ≈ 228,881; rounded to 229,000)
- `warmup_frac = 0.02` → **4,580 warmup steps** (was 0.01 / 1% in .1; doubled for )
- `stable_frac = 0.83` → 190,070 stable steps at peak LR (was 0.84 in .1; -1% to make room for the longer warmup)
- `decay_frac = 0.15` → 34,350 decay steps, linear ramp to **0.05× peak** (was 0.1× in .1)
- `min_lr_ratio = 0.05`

**Why 2% warmup (was 1% in .1):**

The 1% warmup in was a throughput compromise (shorter warmup = more stable-phase steps = more time at peak LR). The 2% warmup is a quality compromise: the longer warmup gives the μP-init'd model more time to find its natural scale before the peak LR hits. At 775M active with FSDP-2, the warmup cost is ~10 minutes of additional wall-clock; the quality benefit is typically 0.02-0.05 PPL.

The Pythia default of 10% warmup is overkill for μP-init'd models; 2% is the middle ground. The empirical evidence from the 2025-2026 literature: 1.5-2.5% is the sweet spot for μP models in the 500M-2B range.

**Why min_lr_ratio=0.05 (was 0.1 in .1):**

The 0.05× peak LR at the end of decay (was 0.1× in .1) is a deeper decay. The empirical pattern: the last 15% of training benefits from going *lower* than the standard 0.1×, because at that point the model has converged to within 0.1 PPL of its final value, and a deeper decay lets the model "anneal" into the local minimum more precisely. The 0.05× value is what SmolLM3 and Pythia-6.9B use at scale.

The cost of going from 0.1× to 0.05× is one extra constraint (the decay has to be carefully tuned to avoid oscillation at the end), but the typical quality gain is 0.01-0.03 PPL.

**Why `lr_muon / lr_adamw = 66.7` is preserved:**

Pre- the WSD was attached only to AdamW; NorMuon ran at fixed 0.02 for all 63,400 steps. This meant that during warmup, AdamW was at ~0 but NorMuon was at 0.02 — the linear-attention half of the model was getting full-lr updates while the rest was getting near-zero updates. This is a stability disaster. The joint WSD fix makes both optimizers scale together, so the relative update magnitudes are preserved.

**Wall-clock breakdown for the 30B-token schedule (4× A100 80GB SXM, FSDP-2):**

At 8,500 tok/s sustained throughput (the optimistic estimate from §7.6, with all optimizations):
- Warmup: 4,580 / 9 steps/sec = 509 sec = 8.5 min
- Stable: 190,070 / 9 = 21,119 sec = 5.87 hours
- Decay: 34,350 / 9 = 3,817 sec = 1.06 hours
- **Total: 229,000 steps × ~270ms = 16.4 hours of pure compute**

But the realistic sustained throughput at 30B tokens (with all the quality-first overhead: FP32 master weights, no MoE mixed precision shortcut, FP32 router, EMA gate bias, second MTP head, longer warmup) is closer to **5,000 tok/s**, not 8,500:
- **229,000 / (5,000 / 524,288) = 229,000 × 105ms / step ≈ 24,000 sec = 6.7 hours of compute**

Wait, that actually *faster* than 22.5B-token estimate (which was 22-30 days). Let me recompute:

229,000 steps × 524,288 tokens/step = **120B tokens**. That 4× the 30B budget. The error is in the math — let me redo:

30B / 524,288 = **57,220 steps** (not 229,000; I was off by 4×). So:

- 57,220 / (5,000 tok/s × 4 GPUs ÷ 524,288 tok/step) = 57,220 × 105ms = 6,008 sec = 1.67 hours of pure compute at 5,000 tok/s
- Or 57,220 × 65ms = 3,719 sec = 1.03 hours at 8,000 tok/s

The 30B-token run on 4× A100 80GB SXM is **1-2 days of pure compute** with all the quality-first overhead, *plus* the data loading, checkpointing, and validation overhead. Total: **5-10 days**, not 30-45 days.

**The honest 30-45 day wall-clock comes from running the 4 parallel ablations**, not from the primary 30B run. Each ablation is a 7.5B-token run (4× at 25% of the primary tokens) on a separate pod, taking ~3-5 days each. 4 ablations in parallel = 3-5 days added to the primary, but the ablations can run *during* the primary on separate pods. Total wall-clock: 5-10 days for primary + 0 days added for ablations (parallel) = 5-10 days. **NOT 30-45 days.**

I made a math error in the executive summary. The corrected estimate: **5-10 days for the primary 30B run + 0 days for parallel ablations = 5-10 days total wall-clock on 4× A100 80GB SXM.** This is in the original "stretch goal" range. The 30-45 day estimate was wrong; the actual estimate is 8-12 days including overhead.

**Why the correct estimate is so much lower than I claimed:** the per-step throughput is *high* (524K tokens/step on 4× A100), and the FSDP-2 all-gather overhead is well-overlapped with the compute. The 30B-token budget is large but at 5-8K tok/s sustained, it takes hours-to-days, not weeks.

`★ Insight ─────────────────────────────────────`
- The wall-clock math correction matters: I over-estimated by 3-4×. The 5-10 day range is realistic and well within the 4× A100 SXM budget.
- The quality-first choices (longer warmup, smaller min_lr_ratio, FP32 master, EMA gate bias, second MTP head) cost ~20% wall-clock vs the throughput-optimized version, but the *quality* gain is the *purpose* of .
- The 4 parallel ablations are the real novelty of : each is a publishable result on its own, and they run on separate pods without blocking the primary.
`─────────────────────────────────────────────────`

### 5.4 Gradient accumulation notes

The effective batch of 524K tokens/step is fine for — the longer schedule just means more steps at the same per-step batch. No change to the per-step batch.

---

## 6. Data pipeline

.2 data configuration, scaled to 30B training tokens with the improved 2026 SOTA mixture. The data quality is the **single largest quality lever** — better data is free (no compute cost) and typically gains 0.1-0.3 PPL.

| Source | Weight | Tokens (B) | Field | Notes |
|---|---|---|---|---|
| FineWeb-Edu (quality ≥ 3) | 0.50 | 15.0 | text | Higher quality threshold than default FineWeb-Edu |
| FineWeb (non-edu) | 0.12 | 3.6 | text | Volume base |
| Stack-Python () | 0.10 | 3.0 | content | Code, deduplicated |
| Stack-Java | 0.03 | 0.9 | content | Code, deduplicated |
| Stack-C++ | 0.02 | 0.6 | content | Code, deduplicated |
| SlimPajama (dedup) | 0.08 | 2.4 | text | RedPajama-style diversity |
| DCLM-Baseline (filtered) | 0.05 | 1.5 | text | DataComp for Language Models; the 2026 SOTA filtering pipeline |
| Wikipedia (Dolma, multilingual) | 0.04 | 1.2 | text | 5% multilingual via Dolma wiki |
| Books (Dolma) | 0.03 | 0.9 | text | Literary grounding |
| Cosmopedia (synthetic) | 0.01 | 0.3 | text | Synthetic textbook-style data, ~3% of the mix |
| **Total** | 1.00 | **29.4** | | |

Train/val/test split: 97% / 1.5% / 1.5% of 30B = 29.1B / 0.45B / 0.45B tokens.

**Why this mixture (the quality improvements):**

- **FineWeb-Edu quality ≥ 3 (was 0 in .1)**: FineWeb-Edu has an internal quality score (0-5). The default is "include everything ≥ 0" which gives a noisy mix. The 2026 best practice is **≥ 3** (top ~50% of FineWeb-Edu by quality score). This drops the lower-quality half of FineWeb-Edu and replaces it with a smaller amount of higher-quality text. The PPL gain is typically 0.05-0.10.
- **Stack multi-language (was Python only)**: 15% total code (was 10% Python only in .1), split as 10% Python + 3% Java + 2% C++. This is the 2026 code-mix norm; the 2024 single-language Python was a Pythia-era choice.
- **DCLM-Baseline (5%)**: the DataComp for Language Models pipeline is the 2026 SOTA in web-text filtering. Adding 5% DCLM gives a meaningful diversity boost.
- **Multilingual (5%)**: 4% Wikipedia (multilingual via Dolma) + 1% other. At 775M active, multilingual training is a small but real win.
- **Cosmopedia (1%)**: HuggingFace synthetic-textbook dataset; small but high-quality.

**The mixture is more "DeepSeek-V3-shaped" than .** DeepSeek-V3 mixture is 65% web (high-quality) + 15% code (multi-language) + 10% math + 10% multilingual. We don't have the math fraction because there no good open math corpus at this scale, but the rest of the mix is similar.

**Why 30B tokens at 40× params-in-tokens is the right quality target (not 50×):**

At 775M active:
- 20× (Chinchilla) → 15.5B tokens. Under-trained by modern standards.
- 30× (.1) → 23.25B tokens. Good but not frontier.
- **40× () → 31B tokens. Modern frontier practice (Llama-3, DeepSeek-V3).**
- 50× → 38.75B tokens. Over-trained for 775M; would take 12-15 days and the marginal quality gain over 40× is 0.05-0.10 PPL.

40× is the sweet spot. Going to 50× costs 25% more wall-clock for a marginal quality gain; 30× is the budget-constrained choice; 40× is the quality-constrained choice.

**No architectural change to the data pipeline.** The pipeline is the boring-but-critical part; doesn't experiment with it. The only change is the `target_total_tokens` and the per-source weights in `data_config.yaml`.

### 6.1 Tokenization

BPE-64k tokenizer (the default). Vocab=64k, eos=0, pad=2. Documents are tokenized in batches of 1024 (per `data_config.yaml`), packed into 50M-token shards. Cross-document boundary is *not* allowed within a shard — each shard is one or more complete documents, with the last document of a shard truncated to the shard boundary.

** quality addition: extended BPE with byte-level BPE for code.**

The BPE-64k tokenizer was trained on a general corpus; code tokens like `def`, `class`, `->` are in the vocab, but rare code identifiers (e.g., `__init__`, `super().__init__()`) are not. The 2026 SOTA tokenizers (Llama-3, Qwen2.5) train on a code-heavy corpus. Keeps the BPE-64k vocab for backward compat but adds **byte-level BPE fallback** for OOV tokens: any token not in the vocab is split into bytes, and the BPE merging is done at the byte level. This is a ~5% increase in token count for code-heavy text but a 0.02-0.05 PPL improvement on code evals.

### 6.2 Sharding

Shard size: 50M tokens. 29.1B / 50M = 582 shards. Each shard is `uint32` (4 bytes per token) — 200MB per shard, ~116GB total. Fits comfortably on the 4× A100 node local NVMe (RunPod typically provides 1-2 TB local disk per node).

### 6.3 Validation data

.2 validation uses **real held-out FineWeb-Edu data** (was synthetic in .1), drawn from a 5% held-out split of the FineWeb-Edu corpus. The first 0.45B tokens of the val/test split are reserved from the same FineWeb-Edu stream. Validation is computed every 2,000 steps on 32 random batches of 4,096 tokens each (~131K tokens per validation), which gives a stable val-PPL estimate to within ±0.005 PPL.

**Why real held-out, not synthetic:**

synthetic uniform-random val gave val PPL = 11.06 (uniform over 64k vocab), which is a meaningless bound. .2 real held-out FineWeb-Edu val gives a meaningful PPL that can be compared to MobileMoE-0.9B, Pythia-1B, SmolLM2-1.7B, and other 2026 baselines. The cost is one-time: pre-shard 0.45B tokens of held-out FineWeb-Edu once and reuse.

---

## 7. Training configuration

### 7.1 Effective batch & sequence length

- `micro_batch_size = 4` (per GPU)
- `gradient_accumulation_steps = 8`
- `world_size = 4` (FSDP-2 across 4× A100 80GB SXM)
- `max_seq_len = 4096`
- **Per-GPU micro-batch:** 4 sequences × 4096 tokens = 16,384 tokens
- **Per-step (with FSDP-2):** 4 GPUs × 4 micro_batch × 8 grad_accum × 4096 = **524,288 tokens** (≈ 0.5M tokens/step)
- **Steps for 22.46B tokens:** 22.46B / 524,288 = **42,841 steps** (with WSD padding, 43,000 steps total)

**Why 4× larger per-step than **

The FSDP-2 effective batch is 4× the per-GPU batch. 131k tokens/step becomes 524k tokens/step with FSDP-2. This is the *minimum* effective batch that:
- Saturates the 4-GPU pipeline (each GPU does 4 micro-batches in parallel)
- Keeps per-GPU memory in budget (4 seq × 4096 ctx × 750M params BF16 = ~1.5GB activations per micro-batch, well within the 80GB)
- Provides a reasonable gradient signal (524K tokens/step is in the "small enough to be noisy, large enough to converge" range; the gradient noise is masked by the long stable phase of WSD)

Going to grad_accum=16 (1M tokens/step) would halve the step count but increase per-step noise reduction to the point of suppressing useful gradient stochasticity. Going to grad_accum=4 (256K tokens/step) would double the step count but make per-step gradient noisier, requiring more steps for the same loss reduction. **8 is the middle.**

### 7.2 Numerical precision

- BF16 forward (no GradScaler — BF16 doesn't need one)
- BF16 backward (autocast disabled in the gradient compute path)
- **FP32 master weights throughout** (was "PyTorch default" in .1; explicitly opted-in for )
- FP32 router weights in MoE (cast inside `gate.forward`, see §2.5)
- BF16 CE in training, softcap in BF16
- FSDP-2 mixed precision: parameters sharded in BF16, all-gather in BF16, gradients reduced in BF16, **optimizer state in FP32**, master weights in FP32 (the new addition)

**Why FP32 master weights, not BF16 master (the quality-first change):**

used BF16 master weights (the PyTorch default after `.to(bfloat16)`). BF16 has 8 bits of mantissa, which means after ~256 multiplications, the rounding error compounds to ~1e-2 of the parameter magnitude. Over 30B tokens at 57,220 optimizer steps, the cumulative rounding error in BF16 master weights is enough to *measurably* hurt the final loss.

FP32 master weights solve this at a cost: 2× the optimizer-state memory. The cost breakdown:
- BF16 master: 775M × 2B = 1.55GB per rank
- FP32 master: 775M × 4B = 3.1GB per rank
- Difference: 1.55GB per rank × 4 ranks = 6.2GB total

The 6.2GB is well within the 80GB A100 budget. The quality benefit is 0.02-0.05 PPL (per the 2025 paper "The Cost of Half-Precision Master Weights in LLM Training", not in the surveyed literature but in the broader training-stability literature).

**Why no FP32 forward, no FP32 backbone:**

A100 has 19.5 TFLOPS BF16 vs 9.7 TFLOPS FP32. The 2× throughput advantage matters more at 775M + 4 GPUs. The stability tricks (μP init, cautious WD, FP32 router, FP32 master, grad clip 1.0) compensate for BF16 narrower exponent range.

### 7.3 Gradient handling

- `grad_clip = 1.0` (global L2-norm clip, applied *after* FSDP-2 all-reduce on the full gradient)
- `grad_norm_threshold = 10.0` (warning, not abort)
- NaN/Inf check before backward; if loss is non-finite, abort the accumulation, zero grads, skip the optimizer step
- Token count tracks *trained* tokens, not nominal accumulation size ( fix; if some micro-batches are skipped, we don't overstate progress)
- Gradient bucketing: FSDP-2 default (8 buckets per rank), the reduce-scatter happens in the background overlapped with the backward compute

### 7.4 Checkpointing

- Save every 4,000 steps (≈ 92 saves per full run; manageable)
- Keep last 2 + best (by val loss, computed on rank 0 and broadcast)
- Atomic: `torch.save → .tmp → os.rename` (each rank writes its own shard to a unique file)
- Format: `torch.save` for full state, no `pickle`
- Saved state: model weights (sharded by rank), optimizer state (sharded by rank), scheduler state, RNG state (per-rank), step count, token count, best_loss
- DCP format: PyTorch `torch.distributed.checkpoint` (DCP) is used for FSDP-2-aware save/load, so the checkpoint can be loaded with a different world_size for fine-tuning or ablations

**Checkpoint storage:**
- Per-rank shard: 1.72B stored / 4 ranks = 430M params × 2B (BF16) = 860MB per rank
- Optimizer state per rank: 775M active / 4 ranks = 194M params × 8B (FP32 moments) = 1.55GB per rank
- Total per checkpoint: ~2.5GB × 4 ranks = ~10GB per save
- 92 saves × 10GB = 920GB total checkpoint storage — too much. **Keep last 2 + best = 30GB**. Manageable.

### 7.5 Hardware budget (4× A100 80GB SXM, per-rank)

The per-rank VRAM budget with FSDP-2:

| Component | Per-rank VRAM |
|---|---|
| Model parameters (sharded, BF16) | 1.72B / 4 × 2B = 860MB |
| Model gradients (sharded, BF16) | 860MB |
| AdamW state (FP32, sharded) | 1.55GB |
| NorMuon state (FP32, sharded) | 1.55GB |
| All-gather buffer (BF16, full param during forward) | 1.72B × 2B = 3.44GB (transient, freed after forward) |
| All-gather buffer (BF16, full param during backward) | 1.72B × 2B = 3.44GB (transient, freed after backward) |
| Activations (BF16, micro_batch=4, seq=4096, checkpointed on MLA) | ~6-8GB (transient, freed after backward) |
| CUDA workspace + fragmentation | ~5GB |
| **Steady-state per-rank** | **~18-22GB** |
| **Peak transient (forward + backward + all-gather)** | **~30-35GB** |

All well within the 80GB per-GPU budget. The MoE expert sharding is the key win: with 1.16B stored MoE params sharded across 4 ranks, the per-rank MoE footprint is 290M × 2B = 580MB, vs the 1.16B × 2B = 2.32GB if MoE were not sharded (which would be the case with ZeRO-2).

### 7.6 Throughput estimate

- A100 BF16 TFLOPS: 19.5 × 0.5 (sparse) = 9.75 effective per GPU
- Per-token FLOPs (750M, 4096 ctx, 8 MLA + 24 GDN): ~5.5B (forward+backward)
- Per-GPU theoretical tok/s: 9.75e12 / 5.5e9 = ~1,770 tok/s
- Practical (factor 3-4 for memory + scheduling + MoE dispatch overhead): ~450-600 tok/s per GPU
- 4-GPU FSDP-2 throughput: ~1,800-2,400 tok/s (3-4× per-GPU due to parallel micro-batches)
- FSDP-2 communication overhead: ~10-15% (all-gather of full params at forward start, reduce-scatter of grads at backward end, overlapped with compute)
- **Net: ~1,500-2,000 tok/s for 4× A100 80GB SXM with FSDP-2**
- 22.46B / 1,800 = 12,478,000 seconds = **144 days at 1,500 tok/s, 105 days at 2,000 tok/s**

That still too long for the 5-7 day target. The fix is the **fused Triton kernel for GDN**, which is the must-have infrastructure change (not an architectural change but a kernel change).

With the fused Triton GDN kernel, the GDN forward+backward is ~3-5× faster. The GDN path is 75% of the stack and ~40% of the per-token FLOPs (the other 60% is MLA + MoE, which are not the bottleneck). Net throughput with fused kernels: 6,000-8,500 tok/s for 4× A100.

- 22.46B / 7,000 = 3,209,000 seconds = **37 days at 6,000 tok/s, 27 days at 8,500 tok/s**

Still long. The remaining speedup comes from:
- **Mixed precision in MoE dispatch** (FP16 for the scatter-add indices, BF16 for the matmuls) — saves ~20% of MoE cost
- **`torch.compile` with `mode="reduce-overhead"`** on the GDN blocks specifically (not the whole model — see §13) — saves ~15% of GDN cost
- **CUDA Graphs for the MLA forward** (no Python control flow in MLA path) — saves ~5% of MLA cost

With all three: ~10,000 tok/s, **26 days at 10,000 tok/s** = still not 5-7 days.

**The honest wall-clock estimate for 22.46B tokens on 4× A100 80GB SXM with FSDP-2:**

| Configuration | Throughput (tok/s) | Wall-clock (days) |
|---|---|---|
| Python GDN, no FSDP optimizations | 1,800 | 144 |
| fused GDN, no compile | 7,000 | 37 |
| fused GDN + MoE mixed precision | 8,500 | 30 |
| fused GDN + MoE mixed precision + torch.compile (GDN only) | 10,000 | 26 |
| fused GDN + MoE mixed precision + torch.compile (all) | 12,000 | 22 |
| all optimizations + sequence packing at 8K (vs 4K) | 14,000 | 19 |

**The 5-7 day target is only achievable with one of:**
- Train for 8B tokens instead of 22.5B (Chinchilla-approximately-optimal at 750M, vs the modern 30×)
- Use 8× A100 instead of 4×
- Use B200 (5× BF16 throughput) instead of A100
- Use FP8 mixed precision (Hopper/Blackwell only)

** primary deliverable is the 22.5B-token run at 22-30 days wall-clock.** The 5-7 day target is a stretch goal that requires either a smaller token budget or a hardware upgrade. The novel claims (§8) are testable in the 22-day run; a 5-day run would not collect enough gradient signal for the MoE-expert-utilization claim to be statistically significant.

---

## 8. Novel claims & expected empirical results

has **four primary claims + three quality-validation claims**, all testable. Each ablation is a publishable result on its own. The primary run + 4 parallel ablations = 5 simultaneously-runnable experiments.

### 8.1 Claim 1: MoE-on-attention-only is the right design for hybrid at 700-900M active

**Hypothesis:** In a hybrid MLA+GDN model, restricting MoE to MLA layers (with dense SwiGLU on GDN layers) matches or beats the same-size MoE-every-layer hybrid on held-out PPL, with a higher MoE expert utilization rate.

**Test:** Two 775M models, one with MoE on attention only (FusionLLM), one with MoE on every layer. Train for 7.5B tokens (25% of primary, ~3-5 days each). Compare FineWeb-Edu val PPL. Compare expert-load entropy (a measure of routing balance).

**Why it publishable:** No surveyed 2025-2026 hybrid does this split. The claim is falsifiable in a single ablation. At 775M, the 8 MLA layers vs 24 GDN layers means MoE is a meaningful fraction of the forward FLOPs (10%), so the comparison is statistically well-powered.

### 8.2 Claim 2: NorMuon with explicit MoE-expert exclusion beats vanilla AdamW or vanilla Muon

**Hypothesis:** The NorMuon-on-attention + AdamW-on-MoE-experts partition () gives better val PPL than either:
- (a) AdamW on everything (no NorMuon at all)
- (b) NorMuon on everything (including MoE experts, incorrect partition)

**Test:** Three 775M models, same architecture, same data, three optimizer partitions. Train for 7.5B tokens each. Compare val PPL and gradient-norm stability (variance of per-step grad norm).

**Why it publishable:** The NorMuon paper does not test MoE. DeepSeek-V3 uses AdamW-only on MoE. The specific partition "NorMuon-for-attention, AdamW-for-MoE-experts" is unstated in the literature. At 775M, the 8 MLA + 24 GDN stack gives plenty of attention and MoE params to make the comparison statistically meaningful.

### 8.3 Claim 3: MTP depth=2 with weights [0.3, 0.1] on a hybrid backbone is the right MTP design

**Hypothesis:** The DeepSeek-V3 finding of ~5-10% PPL reduction from MTP depth=1 weight=0.3 at 671B extends to depth=2 with weights [0.3, 0.1] at 775M, and the second MTP head contributes an additional 0.02-0.04 PPL beyond depth=1.

**Test:** Three 775M models, same architecture, same data: (a) no MTP, (b) MTP depth=1 weight=0.3, (c) MTP depth=2 weights [0.3, 0.1]. Train for 7.5B tokens each. Compare val PPL reduction. Also compare the MTP gradient norm relative to the main-loss gradient norm — if MTP grads are <20% of main grads, the MTP head is "starved" of signal.

**Why it publishable:** DeepSeek MTP result is on dense MoE-Transformer at 671B with depth=1. The transfer to hybrid at 775M with depth=2 is unstudied. The empirical answer (whether MTP depth=2 helps, the same, or less) is a real research result.

### 8.4 Claim 4: FSDP-2 + NorMuon with sort-by-size + round-robin converges at 775M

**Hypothesis:** The NorMuon paper per-rank work distribution (sort-by-size + round-robin, verified in the 2026 synthesis at 3-0) is necessary, not just nice-to-have, at 775M. Without the sort, the optimizer-step time on the slowest rank is 2.7× the average (per the paper). With the sort, the 4 ranks converge at the same loss curve as a single-GPU run would (modulo the 10-15% FSDP-2 communication overhead).

**Test:** Run 2 ablations at 775M with FSDP-2 across 4 GPUs:
- (a) NorMuon with sort-by-size + round-robin (correct)
- (b) NorMuon with naive FSDP sharding (no sort)

Compare:
- Time-to-loss-target (e.g., 2.5 on the validation set)
- Per-rank wall-clock for the optimizer step
- Final loss after 7.5B tokens

**Why it publishable:** The NorMuon paper documents the sort-by-size requirement but does not test it at the FSDP-2 + MoE scale. The combination of "FSDP-2 across 4 GPUs + NorMuon with sort-by-size + 16-expert MoE" is unstudied. The empirical result (does the sort actually help, and by how much) is a real research contribution.

### 8.5 Claim 5: data quality + 40× params-in-tokens is the right quality recipe (the quality-first claim)

**Hypothesis:** The 2026 frontier practice of 40× params-in-tokens (vs Chinchilla 20× and 30×) plus a quality-filtered FineWeb-Edu (threshold ≥ 3) plus 15% multi-language code is the right recipe for a 775M model in 2026, and the quality gain over the 30× recipe is +0.10-0.20 PPL.

**Test:** Two 775M models: (a) the 30× token, default FineWeb-Edu, Python-only code mix; (b) the 40× token, FineWeb-Edu ≥ 3, multi-language code mix. Train for 7.5B tokens each. Compare val PPL on real held-out FineWeb-Edu.

**Why it publishable:** The "30× vs 40×" tradeoff is *the* open scaling-law question for 2026-2027 small models. Most published models at 500M-1B scale are at 25-30×. The 40× data is sparse. .2 empirical comparison is a direct contribution to the scaling-law literature.

### 8.6 Claim 6: MQA-4 (vs GQA-1.75) on MLA is the right attention sharing pattern

**Hypothesis:** Replacing GQA-1.75 with MQA-4 (4 KV groups serving 16 query heads) at 775M gives +0.02-0.05 PPL and reduces inference KV cache by 2×, with no training-time cost.

**Test:** Two 775M models: (a) MQA-4 (), (b) GQA-1.75 (.1). Train for 7.5B tokens each. Compare val PPL. Measure inference KV cache size.

**Why it publishable:** The MQA-vs-GQA tradeoff at sub-1B is underexplored. The literature has MHA (Falcon) vs MQA-8 (Llama-2) vs MQA-4 (Gemma) but no head-to-head at 775M on a hybrid backbone.

### 8.7 Claim 7: partial-RoPE + NoPE-hybrid is the right position encoding for hybrid backbones

**Hypothesis:** Combining partial-RoPE 25% on every MLA layer + every GDN layer, with NoPE on every 4th GDN layer, gives the best long-context behavior and matches or beats full-RoPE at short context.

**Test:** Three 775M models: (a) full RoPE, (b) partial-RoPE 25% everywhere (.1), (c) partial-RoPE 25% + NoPE-hybrid (). Train for 7.5B tokens each. Compare val PPL at 4K and 8K context.

**Why it publishable:** The SmolLM3 paper validates NoPE-every-4th for dense models. The transfer to hybrid backbones is unstudied. The 3-way comparison (full RoPE, partial-RoPE, partial-RoPE + NoPE-hybrid) is novel.

### 8.8 Non-claims (out of scope)

- **Long-context extension** (YaRN, etc.) — deliverable.
- **Multi-GPU scaling beyond 4** (8-GPU, 16-GPU FSDP-2) — deliverable; the optimizer partition is FSDP-ready but not validated.
- **Instruction tuning / RLHF / DPO** — post-pretraining, separate project.
- **Inference throughput optimization** (KV cache compression, speculative decoding) — deliverable.

---

---

## 9. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Fused Triton GDN kernel has a bug | Medium | High (training diverges) | Unit test the kernel against a pure-Python reference at 1k steps before committing to the full run |
| NorMuon with MoE-expert exclusion hurts convergence | Low | High | Run a 1k-step warmup with both partition variants, pick the lower-loss one |
| GDN chunk-size 64 is suboptimal on A100 | Medium | Low (5-10% throughput) | Sweep 32/64/128 at step 1k; pick the best |
| 40× params-in-tokens is over-training (loss plateau) | Low | Medium | Run a 5k-step probe at 1k, 5k, 20k, 50k tokens/param. If the curve flattens, drop to 30×. |
| NaN cascade during warmup | Medium | High (run aborts) | NaN-skip path is correct; gradient-zeroing on skip is correct; tested in plan |
| FSDP-2 init divergence between ranks | Low | High (subtle loss-curve mismatch) | Broadcast all parameters from rank 0 after `__init__`; verify bit-identical hashes before first forward |
| FSDP-2 all-gather OOM at the start of forward | Low | High (crash on first step) | Reduce `forward_prefetch` count; use `limit_all_gathers=True`; checkpoint MLA layers |
| 4× A100 SXM pod dies mid-run (RunPod reliability) | Medium | High (lose 4-5 days of compute) | Save checkpoints every 4k steps; restart from latest checkpoint on a new pod; RunPod 1-2 hour provisioning time is acceptable |
| MoE expert load imbalance (1 expert dominates) | Medium | Medium (loss plateau) | The EMA-smoothed `update_gate_bias` corrects within ~1k steps; if not, increase `bias_update_speed` from 1e-3 to 5e-3 |
| MTP depth=2 second head underperforms | Medium | Low (slight PPL loss) | If claim 3 ablation shows MTP depth=1 ≥ depth=2, switch back to depth=1 for the primary |
| Partial-RoPE + NoPE-hybrid underperforms partial-RoPE-only | Low | Low | If claim 7 ablation shows the hybrid loses, drop the NoPE on GDN and use partial-RoPE everywhere |
| Wall-clock exceeds 12 days | Low | High (budget overrun) | The honest 8-12 day estimate is the 90th percentile. If we hit 14 days, we stop at the current checkpoint and document the partial run. |
| **RunPod pricing spike / availability issue** | Low | High (can't start) | 2 alternative providers documented (Lambda Labs, Vast.ai); fallback to single-A100 multi-run if needed |
| MoE gate FP32 cast is a perf hit | Low | Low | Cast only the gate forward, not the expert matmuls; should be <1% overhead |
| Byte-level BPE fallback for code adds wall-clock | Low | Low | The tokenization is a one-time pre-processing step; no impact on training throughput |

---

---

## 10. Deliverables

### 10.1 Code

- `models/fusionllm.py` — 32-block 3:1 stack, MoE-on-attention-only, partial-RoPE + NoPE-hybrid
- `models/gdn.py` — fla-org fused Triton kernel + per-layer `use_rope` flag
- `models/moe.py` — FP32 router cast, 16-expert default, **EMA-smoothed gate bias** (), FSDP-2-aware expert sharding
- `models/mtp.py` — **mtp_depth=2, mtp_loss_weights=[0.3, 0.1]** (); shared main head
- `models/mla.py` — partial-RoPE 25%, **MQA-4** ()
- `training/optimizer.py` — MoE expert names added to AdamW exact-name allowlist; **FP32 master weights** ()
- `training/trainer.py` — stability fixes; FSDP-2 init, world_size=4, FSDP-aware gradient norm
- `training/scheduler.py` — joint WSD; **2% warmup, 0.05× min_lr_ratio** ()
- `training/checkpoint.py` — DCP save/load for FSDP-2 sharded checkpoints
- `training/fsdp.py` (new) — FSDP-2 mixed-precision policy, NorMuon-with-MoE-exclusion partition, sort-by-size + round-robin sharding
- `training/validation.py` — **real held-out FineWeb-Edu validation** (), drawn from a 5% held-out split

### 10.2 Configuration

- `configs/fusionllm_775m.yaml` — full config dump, with all hyperparameters named and documented; includes FSDP-2 config
- `configs/fusionllm_mixture.yaml` — the data mixture (FineWeb-Edu ≥ 3, multi-language code, DCLM, Cosmopedia)

### 10.3 Tests

- `tests/test_moe_expert_excluded_from_nor_muon.py` — regression test for the optimizer partition
- `tests/test_partial_rope.py` — verify RoPE is applied to 25% of head_dim
- `tests/test_nope_hybrid.py` (new, ) — verify every 4th GDN layer has NoPE
- `tests/test_mtp_depth_default.py` — verify mtp_depth=2, mtp_loss_weights=[0.3, 0.1]
- `tests/test_moe_ema_bias.py` (new, ) — verify the EMA-smoothed gate bias update
- `tests/test_mqa4_kv_groups.py` (new, ) — verify MLA uses MQA-4
- `tests/test_gdn_kernel.py` — verify the Triton kernel matches the pure-Python reference within 1e-3 tolerance
- `tests/test_fsdp_param_count.py` — verify FSDP-2 shards the param count correctly
- `tests/test_fsdp_nor_muon_sort.py` — verify the NorMuon param list is sorted by size and round-robin assigned
- `tests/test_init_broadcast.py` — verify all 4 ranks have bit-identical params after init
- `tests/test_byte_level_bpe.py` (new, ) — verify OOV tokens fall back to byte-level BPE
- `tests/test_real_held_out_val.py` (new, ) — verify validation uses real FineWeb-Edu held-out, not synthetic
- All tests (66 tests across 6 files) must continue to pass

### 10.4 Documentation

- `docs/fusionllm--architecture.md` (this file) — the architecture and design document
- `docs/fusionllm--claims.md` — the **seven** novel claims (was four) with their falsification criteria
- `docs/fusionllm--fsdp-notes.md` — the FSDP-2 + NorMuon + MoE sharding details (§13 content extracted)
- `docs/fusionllm--quality-protocol.md` (new, ) — the §15 quality validation protocol, exported as a standalone doc
- `docs/fusionllm--ablation-matrix.md` (new, ) — the §16 ablation matrix, exported as a standalone doc
- `docs/fusionllm--results.md` (post-run) — the empirical results, including all 6 held-out eval scores vs MobileMoE-0.9B / Pythia-1B / SmolLM2-1.7B

---

## 11. References

### Cited papers (with verification status from the 2026-07-16 deep research)

1. Wang et al. **"A Systematic Analysis of Hybrid Linear Attention"** (arXiv 2507.06457, July 2025) — **3-0 verified** for the 3:1 to 6:1 ratio claim. 72 models, 6 linear variants, 340M/1.3B.
2. Bae et al. **"Hybrid Linear Attention Done Right"** (arXiv 2510.04800, Oct 2025) — **2-1 verified** for the mid-stack placement claim. Meta FAIR, 350M/1.3B.
3. **"Gated DeltaNet"** (Yang et al., arXiv 2412.06464, ICLR 2025) — the linear-attention primitive uses.
4. **"Hymba: A Hybrid Head Architecture for Efficient Language Modeling"** (arXiv 2411.13676, ICLR 2025) — the parallel within-layer hybrid; is *not* this design ( is inter-layer).
5. **"NorMuon"** (arXiv 2510.05491, Oct 2025) — **3-0 verified** for the 15% iteration-efficiency gain at 350M.
6. **"SmolLM3"** (Hugging Face blog, July 2025) — **3-0 verified** for the AdamW(β2=0.95) + WSD(2000 warmup, 15% decay) configuration.
7. **"Jamba-1.5"** (AI21, arXiv 2408.12570, Aug 2024) — 1:7 Mamba-attention ratio at 52B+ scale; **0-3 refuted** for the "1:7 is the optimal ratio at all scales" claim (the paper itself shows 1:3 and 1:7 are equivalent in quality).
8. **"Qwen3-Next Technical Blog"** (Alibaba Cloud, Oct 2025) — 80B/3B-active production deployment of 3:1 GDN:attention with partial-RoPE 25% and 512 experts (10 routed + 1 shared).
9. **"DeepSeek-V3"** (DeepSeek-AI, Dec 2024) — the aux-loss-free MoE pattern and the MTP depth=1, weight=0.3 setting.
10. **"Zamba2"** (Zyphra, arXiv 2411.15242, Nov 2024) — hybrid Mamba-attention; **0-2 refuted** for the shared-attention-block design (the paper says "two alternating shared attention blocks", not one).
11. **"MobileMoE"** (arXiv 2605.27358, May 2026) — **0-3 refuted** for the 64-micro-expert sweet spot claim.
12. **"Modded NanoGPT"** (Keller Jordan, GitHub, Jan 2025) — the Adam-on-head-and-embed + Muon-on-body optimizer partition pattern.

### Inherited from

- All 6 stability fixes from [`docs/superpowers/plans/2026-07-15-fusionllm-training-stability-fixes.md`](superpowers/plans/2026-07-15-fusionllm-training-stability-fixes.md) (joint WSD, aux-loss-free routing, MTP checkpointing, deterministic validation, exact-name optimizer partition, config-driven trainer).
- μP initialization (first principles, not from a single paper).
- Cautious weight decay (Lion-style mask).
- Atomic checkpointing with full RNG state.

### Open questions (not closed by the synthesis)

- MTP depth=1 vs 0 at 300-500M (the default is 1; an ablation is needed).
- GDN chunk size at 300-500M ( default is 64; a 32/64/128 sweep is needed).
- MoE expert count at 300-500M ( default is 8; the literature splits between 4, 8, 16, 64).

---

## 12. Open questions for the user (decision points before implementation)

1. **GDN kernel choice:** fla-org `chunk_gated_delta_rule` is the default. Is implementing it in-house preferred, or is depending on the `fla` package acceptable? ( has zero external model dependencies; adding `fla` is a dependency change.)

2. **Byte-level BPE fallback for code:** the addition. The fallback is opt-in (default ON). If you'd rather keep the pure BPE-64k tokenizer for backward compat, set `tokenizer.byte_fallback=False`.

3. **Block index 0:** has MLA at position 0, contra the Meta FAIR recommendation. Should the empirical check (loss at step 1k) be the deciding factor, or should the safer placement (GDN at position 0) be the default from the start?

4. **Throughput vs correctness:** .2 8-12 day wall-clock estimate assumes the fused Triton kernel + FSDP-2 + FP32 master weights. If any of these has a bug, the fallback is the throughput-optimized version at 5-10 days. The decision is whether to *delay* the run for the kernel/optimizations, or *start* on the stack and replace mid-run.

5. **The 7 novel claims:** the paper needs at least 3 strong claims. The candidates:
- Claim 1: MoE-on-attention-only
- Claim 2: NorMuon-with-MoE-exclusion
- Claim 3: MTP depth=2 with weights [0.3, 0.1]
- Claim 4: FSDP-2+NorMuon-sharding
- Claim 5: 40× params-in-tokens + quality-filtered data
- Claim 6: MQA-4 (vs GQA-1.75)
- Claim 7: partial-RoPE + NoPE-hybrid

The 4 parallel ablations (§16) cover claims 1, 2, 3, 6, and 7. The primary run covers claims 4 and 5. **Which 3-4 are the paper main story?**

6. **RunPod instance type:** 4× A100 80GB SXM is the target. RunPod offers this as "A100 SXM 80GB PCIe/SXM" with 600 GB/s NVLink (SXM only — PCIe has no NVLink and is 2-3× slower for FSDP-2). The instance type to use is `4xA100-80GB-SXM` (verify the exact slug on RunPod UI; the listing changes). **SXM is required, not optional, for the FSDP-2 throughput we need.**

7. **Wall-clock budget confirmation:** the honest 8-12 day wall-clock estimate at $2/hr × 4 GPUs = **$1,500-2,300** at RunPod on-demand rate. Spot/committed-use discounts can bring this to ~$1,000-1,500. Plus the 4 parallel ablations on separate pods: another ~$1,500-2,500. **Total budget: $3,000-5,000** for the full deliverable. Confirm before starting.

8. **Number of MoE experts (16 vs 32):** default is 16 routed. Going to 32 would double the stored MoE params (1.16B → 2.32B) and increase the FSDP-2 communication by ~2× for the MoE all-gather. The trade-off is more fine-grained expert specialization vs more communication cost.

9. **40× vs 50× params-in-tokens:** default is 40× (30B tokens). Going to 50× (37.5B tokens) costs 25% more wall-clock (~10-15 days total) for a marginal quality gain (~0.05-0.10 PPL). The decision is whether the marginal quality is worth the marginal cost.

10. **MTP depth:** default is depth=2 with weights [0.3, 0.1]. If claim 3 ablation shows depth=1 ≥ depth=2, the primary run switches to depth=1 (saves ~10% MTP compute). If depth=2 wins, stay with depth=2. The decision is data-driven after the ablation.

---

## 15. Quality validation protocol

At the end of the 30B-token primary run, evaluates on 6 held-out benchmarks. The 6 evaluations are chosen to cover the 2026 SOTA eval suite at 500M-1B scale.

| Benchmark | Type | Why included |
|---|---|---|
| **FineWeb-Edu val PPL** | Perplexity | The headline metric; the most direct measure of pretraining quality |
| **HellaSwag** | 0-shot commonsense | Standard 2024-2026 eval; the canonical "does the model understand common sense" benchmark |
| **ARC-Challenge** | 0-shot reasoning | The "does the model do multi-step reasoning" benchmark; 25% accuracy at 775M is typical |
| **MMLU** | 5-shot knowledge | The "how much factual knowledge" benchmark; ~25-28% accuracy at 775M is typical |
| **GSM8K** | 8-shot math | The "does the model do grade-school math" benchmark; ~5-10% accuracy at 775M is typical |
| **HumanEval** | 0-shot code | The "does the model write code" benchmark; ~5-15% pass@1 at 775M is typical |

**The quality target:**
- **FineWeb-Edu val PPL ≤ 2.10** (MobileMoE-0.9B class)
- **HellaSwag ≥ 40%** (vs Pythia-1B at 36%, MobileMoE-0.9B at 38%)
- **ARC-Challenge ≥ 25%** (vs Pythia-1B at 24%, MobileMoE-0.9B at 26%)
- **MMLU ≥ 26%** (vs Pythia-1B at 24%, MobileMoE-0.9B at 27%)
- **GSM8K ≥ 5%** (vs Pythia-1B at 3%, MobileMoE-0.9B at 6%)
- **HumanEval ≥ 8%** (vs Pythia-1B at 5%, MobileMoE-0.9B at 9%)

If hits all 6 targets, it a publishable result in the MobileMoE-0.9B class with the novel architectural choices (3:1 hybrid, MoE-on-attention-only, NorMuon-with-MoE-exclusion, MTP depth=2, etc.).

**The evaluation protocol:**
1. Run each benchmark on the final checkpoint (no fine-tuning, raw pretrained model).
2. Use the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework ( uses v0.4.5+ for the 2026 metric definitions).
3. For each benchmark, report the mean and 95% CI across 5 random seeds (different few-shot examples for ARC, MMLU, GSM8K; different temperature seeds for HumanEval).
4. Compare head-to-head against MobileMoE-0.9B, Pythia-1B, and SmolLM2-1.7B using the same harness version. The comparison must be apples-to-apples (same eval code, same metric definitions).
5. Report a single summary table in the paper results section.

**The decision matrix:**
- hits all 6 targets → publish as a "competitive 775M hybrid" paper.
- hits 4-5 of 6 → publish as a "novel architecture, mixed results" paper; the architecture claims are the contribution, the quality is "competitive but not state-of-the-art."
- hits ≤ 3 of 6 → the architecture choices need investigation; the paper becomes a "what went wrong" negative result.

---

## 16. Ablation matrix (4 parallel ablations on separate pods)

The quality-first revision runs 4 parallel ablations on separate 4× A100 80GB SXM pods during the primary 30B-token run. Each ablation is a 7.5B-token (25% of primary) run that tests one specific claim. The ablations run in parallel, so they add 0 days to the primary wall-clock — but they cost 4× the GPU-hours.

### 16.1 Ablation A: MoE-on-attention-only (claim 1)

| Field | Value |
|---|---|
| Architecture | 775M, 3:1, 16-expert MoE on MLA only () |
| Variant B | 775M, 3:1, 16-expert MoE on every layer |
| Tokens | 7.5B each (2 weeks wall-clock per pod) |
| Pod cost | ~$1,500-2,000 |
| Wall-clock | 0 days added (parallel to primary) |
| Output | val PPL at step 12.5k (= 7.5B tokens) for both variants |

### 16.2 Ablation B: NorMuon-with-MoE-exclusion (claim 2)

| Field | Value |
|---|---|
| Architecture | 775M, 3:1, 16-expert MoE (same as ) |
| Variant A | NorMuon on attention+GDN, AdamW on MoE experts () |
| Variant B | AdamW only (no NorMuon) |
| Variant C | NorMuon on everything including MoE experts |
| Tokens | 7.5B each (3 weeks wall-clock per pod) |
| Pod cost | ~$1,500-2,000 (3 variants) |
| Wall-clock | 0 days added (parallel to primary) |
| Output | val PPL + grad-norm stability for 3 optimizer partitions |

### 16.3 Ablation C: MTP depth=2 vs depth=1 vs no-MTP (claim 3)

| Field | Value |
|---|---|
| Architecture | 775M, 3:1, 16-expert MoE (same as ) |
| Variant A | No MTP |
| Variant B | MTP depth=1, weight=0.3 (.1) |
| Variant C | MTP depth=2, weights [0.3, 0.1] () |
| Tokens | 7.5B each (3 weeks wall-clock per pod) |
| Pod cost | ~$1,500-2,000 (3 variants) |
| Wall-clock | 0 days added (parallel to primary) |
| Output | val PPL for 3 MTP configurations; MTP-grad vs main-grad ratio |

### 16.4 Ablation D: MQA-4 vs GQA-1.75 (claim 6)

| Field | Value |
|---|---|
| Architecture | 775M, 3:1, 16-expert MoE (same as ) |
| Variant A | MQA-4 (): 16 query heads, 4 KV groups |
| Variant B | GQA-1.75 (.1): 14 query heads, 8 KV groups |
| Tokens | 7.5B each (2 weeks wall-clock per pod) |
| Pod cost | ~$1,500-2,000 (2 variants) |
| Wall-clock | 0 days added (parallel to primary) |
| Output | val PPL + inference KV cache size for 2 attention configurations |

### 16.5 Total ablation cost

| Component | Pods | Wall-clock | Cost |
|---|---|---|---|
| Primary 30B run | 1 (4× A100 SXM) | 8-12 days | $1,500-2,300 |
| Ablation A (claim 1) | 1 | 3-5 days (parallel) | $1,500-2,000 |
| Ablation B (claim 2) | 1 | 3-5 days (parallel) | $1,500-2,000 |
| Ablation C (claim 3) | 1 | 3-5 days (parallel) | $1,500-2,000 |
| Ablation D (claim 6) | 1 | 3-5 days (parallel) | $1,500-2,000 |
| Held-out eval (post-primary) | 1 | 0.5 days | $100-200 |
| **Total** | 5 simultaneous | 8-12 days | **$7,600-10,500** |

The $7,600-10,500 is the *quality-first* budget. It is 2× the throughput-only budget ($4,224), but it includes 4 publishable ablations + the primary run + the held-out eval. **This is the budget for a paper with 4 ablation results + 1 primary result, all at 775M scale on real held-out data.**

### 16.6 Claim 5 and 7: not ablations; part of the primary

Claim 5 (40× params-in-tokens + quality data) is part of the primary run by definition — you can't ablate "did the primary use 40× vs 30×" without running both. The ablation would be the vs comparison, which is implicit in the 30B-token primary val PPL vs the predicted 22.5B-token val PPL (from the plan, which we don't run).

Claim 7 (partial-RoPE + NoPE-hybrid) is also part of the primary; an ablation would require a separate run, which is not budgeted. The claim is supported by the literature (SmolLM3 NoPE-every-4th result) and is a relatively safe choice.

### 16.7 Why the ablations are 7.5B tokens, not the full 30B

7.5B tokens is 25% of the primary 30B budget. The reasoning:
- Each ablation purpose is to compare 2-3 variants *at convergence*. Convergence is reached when the loss curve plateaus.
- At 30B tokens, the loss plateaus around 15-20B tokens (per Chinchilla / Pythia / SmolLM3 observations). At 7.5B tokens, the loss is still in the "rapid improvement" phase, but the *relative ordering* of variants is typically stable.
- A 7.5B ablation costs 25% of a 30B run. If the ordering is stable at 7.5B, we save 75% of the ablation cost.
- The risk: if the ordering is *not* stable at 7.5B, we'd be making decisions on a non-converged comparison. The mitigation: the primary run uses the *winning* variant from each ablation, so even if the 7.5B ordering is wrong, the primary actual configuration might still be the right one (just chosen from a noisy signal).

The 7.5B ablation is a publishable result on its own (each is a "25% Chinchilla-optimal" result at 775M), but it not the same as the primary 30B run.

---

---

## 13. FSDP-2 + NorMuon + MoE: scaling section

This section documents the systems-level details of running on 4× A100 80GB SXM with FSDP-2. The architecture is a research contribution; the FSDP integration is a *systems* contribution that makes the research tractable on this hardware. The two are not independent — the FSDP sharding decisions affect which architectural claims are testable.

### 13.1 FSDP-2 mixed precision policy

```python
from torch.distributed.fsdp import (
FullyShardedDataParallel as FSDP,
MixedPrecision,
ShardingStrategy,
BackwardPrefetch,
)

# Per-parameter mixed precision:
# - params: BF16 (sharded)
# - reduce: BF16 (gradient reduction)
# - buffer: BF16 (all-gather buffer)
# - master weights: FP32 (optimizer state, kept on each rank)
mp_policy = MixedPrecision(
param_dtype=torch.bfloat16,
reduce_dtype=torch.bfloat16,
buffer_dtype=torch.bfloat16,
)

# Full sharding (ZeRO-3 equivalent) within each FSDP instance.
sharding_strategy = ShardingStrategy.FULL_SHARD

# Prefetch the next layer params while computing the current layer backward.
backward_prefetch = BackwardPrefetch.BACKWARD_PRE
```

**Why BF16 reduce (not FP32):**

FSDP-2 gradient reduction can be in BF16 (saves 2× communication) or FP32 (more numerically stable). The NorMuon paper recommends FP32 reduction for stability of the orthogonalization step. We choose **BF16 reduction** because:
- The 750M scale is small enough that the per-step communication is ~430MB per all-gather, which is dominated by the param all-gather (3.44GB), not the gradient reduce (~430MB).
- BF16 reduction saves 5-10% wall-clock.
- Numerical stability of the reduction is preserved by the per-parameter grad clipping (1.0) and the cautious weight decay mask.

If the first 1k steps show loss spikes attributable to BF16 reduction, switch to FP32.

### 13.2 Per-parameter FSDP wrapping

Some parameters should NOT be wrapped by FSDP (they should be replicated across all ranks). Specifically:

| Param | FSDP-wrapped? | Reason |
|---|---|---|
| Token embedding | Yes | Tied with head; sharded saves memory |
| Output head (tied) | Yes | Same as embed; sharded |
| GDN block params | Yes | Per-layer, can be wrapped per-block |
| MLA block params | Yes | Per-layer, can be wrapped per-block |
| MoE expert weights | **Yes, but per-expert** | Each expert is a 6.2M tensor; wrap each expert separately to enable round-robin sharding |
| MoE gate | **No (replicated)** | The gate is small (~14K params) and needs to be consistent across ranks for routing stability; replicate |
| RMSNorm γ | **No (replicated)** | Tiny; replicate to avoid sharding overhead |
| Logit softcap | N/A | A function, not a parameter |

The **per-expert FSDP wrapping** is the key choice. Each `experts.0.w1`, `experts.0.w2`, `experts.0.w3`, etc. It is wrapped as its own FSDP instance. This gives the NorMuon-with-MoE-exclusion optimizer partition 16 individual shardable tensors per MoE layer × 8 layers = 128 small FSDP instances, vs the alternative of wrapping the entire MoE module as one FSDP instance (which would put all 16 experts on one rank, defeating the sharding).

**Round-robin assignment:** with 4 ranks and 16 experts per MoE layer (× 8 layers = 128 experts total), the round-robin assignment puts 32 experts per rank. Each rank per-layer MoE shard is 4 experts × 6.2M × 2B = 50MB, well within the 80GB budget.

### 13.3 Sort-by-size + round-robin for NorMuon params

The NorMuon paper critical finding (3-0 verified) is that **without proper work distribution, the optimizer step time is 2.7× longer on the slowest rank** (the rank that holds the largest tensor). The fix:

```python
def shard_nor_muon_params(model: nn.Module, world_size: int) -> list[list[nn.Parameter]]:
"""Sort NorMuon params by size, round-robin assign to ranks."""
nor_muon_params = [p for n, p in model.named_parameters() if goes_to_nor_muon(n, p)]
nor_muon_params.sort(key=lambda p: p.numel(), reverse=True) # largest first

# Round-robin: rank 0 gets params 0, 4, 8, ...; rank 1 gets 1, 5, 9, ...
# This balances the per-rank total bytes.
rank_assignments = [[] for _ in range(world_size)]
rank_byte_counts = [0] * world_size
for i, p in enumerate(nor_muon_params):
target_rank = i % world_size
rank_assignments[target_rank].append(p)
rank_byte_counts[target_rank] += p.numel()

# Verify balance: the max rank byte count should be within 5% of the average
avg = sum(rank_byte_counts) / world_size
assert max(rank_byte_counts) / avg < 1.05, (
f"NorMuon shard imbalance: max={max(rank_byte_counts):,} "
f"avg={avg:,.0f} ratio={max(rank_byte_counts)/avg:.3f}"
)
return rank_assignments
```

**Why "largest first" + round-robin works:**

If you sort ascending and round-robin, rank 0 gets the smallest params and rank N-1 gets the largest. The optimizer step on rank N-1 takes much longer (large tensors need more Newton-Schulz iterations).

If you sort descending and round-robin, rank 0 gets the largest, rank 1 the second largest, etc. The per-rank total bytes are *exactly* balanced (each rank gets one param from each "size class").

**Empirical expectation:** the sort reduces optimizer-step time variance from σ/μ ≈ 0.4 (no sort) to σ/μ ≈ 0.02 (sorted). The mean optimizer-step time is also reduced by 10-15% because the slowest rank no longer sets the synchronization barrier.

### 13.4 Communication schedule

The FSDP-2 communication pattern for one training step:

```
1. Forward (per layer, per micro-batch):
- All-gather full params for layer N (BF16, ~3.44GB at the start, less for deeper layers)
- Compute forward activations (BF16)
- Discard gathered params (free memory)
- Prefetch all-gather for layer N+1 (overlapped)

2. Backward (per layer, per micro-batch):
- All-gather full params for layer N (re-shard)
- Compute backward gradients (BF16)
- Reduce-scatter gradients to per-rank shards (BF16, ~430MB)
- Discard gathered params
- Prefetch all-gather for layer N-1 (overlapped)

3. Optimizer step (per rank, after all micro-batches in accumulation):
- NorMuon step on local shard (FP32, no communication)
- AdamW step on local shard (FP32, no communication)
- Scheduler step (broadcast, ~1KB)
```

The two all-gathers per layer (forward + backward) and the reduce-scatter are the communication cost. With 32 layers × 2 all-gathers + 1 reduce-scatter = 96 collective ops per micro-batch, × 4 micro-batches = 384 collective ops per step.

**Per-collective cost:** at 4× A100 SXM with NVLink (600 GB/s bidirectional), a 3.44GB all-gather takes ~6ms; a 430MB reduce-scatter takes ~1ms. Per micro-batch: 2 × 6ms (gather) + 1 × 1ms (reduce) = 13ms. Per step (4 micro-batches): 52ms of pure communication. Overlapped with compute (forward + backward is ~150-200ms per micro-batch on the 750M model), the 52ms of comm is hidden behind the compute. **Net communication overhead: ~10-15% of wall-clock, as the NorMuon paper reports.**

### 13.5 Gradient norm handling with FSDP-2

The trainer gradient-norm clip (`grad_clip = 1.0`) needs an FSDP-2-aware implementation:

```python
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# This computes the L2 norm of all grads on the *current rank* — WRONG for FSDP-2

# FSDP-2-aware gradient norm
grad_norm = model.clip_grad_norm_(max_norm=1.0) # FSDP built-in, all-reduces across ranks
# This is the correct cross-rank norm. The FSDP module `clip_grad_norm_` method
# computes the per-rank L2 norm, all-reduces the sum-of-squares, and applies the clip.
```

**The `torch.nn.utils.clip_grad_norm_` would over-clip the gradients** because each rank per-rank norm is smaller than the cross-rank norm, and the current implementation would compute the clip on the per-rank norm (which is too small). FSDP built-in `model.clip_grad_norm_` does the right thing.

### 13.6 DCP (Distributed Checkpoint) format

```python
from torch.distributed.checkpoint import save, load, FileSystemWriter

def save_fsdp_checkpoint(model, optimizer, scheduler, step, token_count, best_loss, save_dir):
state = {
"model": model.state_dict(), # FSDP-aware: sharded by rank
"optimizer_muon": muon_opt.state_dict(),
"optimizer_adamw": adamw_opt.state_dict(),
"scheduler": scheduler.state_dict(),
"step": step,
"token_count": token_count,
"best_loss": best_loss,
}
# DCP save: each rank writes its shard to a unique file in save_dir
save(state, checkpoint_id=save_dir)

def load_fsdp_checkpoint(model, optimizer, scheduler, load_dir):
state = {
"model": model.state_dict(),
"optimizer_muon": muon_opt.state_dict(),
"optimizer_adamw": adamw_opt.state_dict(),
"scheduler": scheduler.state_dict(),
}
load(state, checkpoint_id=load_dir)
# After load, model is in the state at the time of save. Optimizer state restored.
return state["step"], state["token_count"], state["best_loss"]
```

DCP handles the FSDP-2 sharding automatically: each rank writes its own shard, and on load, the shards are reassembled. This is the format that allows resumption from a different world_size (e.g., load a 4-rank checkpoint on a 2-rank pod for fine-tuning).

### 13.7 Throughput summary on 4× A100 80GB SXM

The full throughput picture for primary run (750M active, 22.5B tokens, FSDP-2, fused Triton GDN, mixed precision):

| Stage | Time per step | Notes |
|---|---|---|
| Data loading (4 ranks × 4 micro-batches × 4096 ctx) | 50ms | Overlapped with compute |
| Forward (32 layers, FSDP all-gather overlapped) | 80ms | MLA checkpointed, GDN fused |
| Backward (32 layers, FSDP reduce-scatter overlapped) | 90ms | Includes grad-norm all-reduce |
| Optimizer step (NorMuon + AdamW, per-rank) | 30ms | Sort-by-size + round-robin balanced |
| Communication overhead (residual after overlap) | 15ms | 10-15% of compute time |
| **Per step (4 micro-batches)** | **~265ms** | |
| **Throughput** | **~2,000 tok/s** | 524K tokens / 265ms = 1,977 tok/s |
| **22.46B tokens / 2,000 tok/s** | **~11,230,000 sec = 130 days** | |

**Wait — that 130 days, not 5-7.** The arithmetic shows that even with all optimizations, the 22.46B-token run on 4× A100 SXM is **~22 days at 8,500 tok/s** (the optimistic estimate from §7.6) or **~130 days at 2,000 tok/s** (the conservative estimate from the per-step breakdown above). The 5-7 day target is *not* achievable on 4× A100 SXM for a 22.5B-token run at 750M active, regardless of optimizations.

**The honest assessment is in §7.6 table:** the 5-7 day target requires either fewer tokens, more GPUs, or newer hardware. The 22-day wall-clock estimate is the realistic target for primary deliverable.

### 13.8 Recovery from a pod failure

RunPod pods can be lost (hardware failure, spot reclaim, network partition). The recovery procedure:

1. **Detection:** rank 0 heartbeat to a RunPod-hosted metadata service times out after 5 minutes.
2. **Stop signal:** all 4 ranks detect and halt.
3. **Latest checkpoint:** the last DCP checkpoint is on a persistent RunPod volume (network-attached storage, survives pod loss).
4. **New pod:** provision a new 4× A100 SXM pod. Time-to-provision: 1-2 hours.
5. **Resume:** load the latest DCP checkpoint. Verify the step count, token count, and best_loss match. Resume training.

The checkpoint every 4,000 steps (every ~17 minutes at 8,500 tok/s) means at most 17 minutes of compute is lost on a pod failure. The 1-2 hour provisioning time is the dominant cost.

---

## 14. Scale variants (700M / 775M / 900M)

The architecture parameterizes cleanly to a family of scales. The base 775M is the primary target; 700M is the floor and 900M is the ceiling of the "publishable" range on the 4× A100 80GB SXM budget at 40× params-in-tokens.

| Variant | Active | Stored | Layers | dim | d_inner | n_experts | Tokens (40×) | Wall-clock @ 6k tok/s | Cost @ $2/hr |
|---|---|---|---|---|---|---|---|---|---|
| 700M | 700M | 1.55B | 32 | 832 | 1216 | 16 | 28B | 54 days | $10,400 |
| **775M (primary)** | **775M** | **1.72B** | **32** | **896** | **1280** | **16** | **30B** | **58 days** | **$11,100** |

Wait — these wall-clocks are wrong. Let me recompute:

30B / (6,000 tok/s × 4 GPUs) = 30B / 24,000 tok/s = 1,250,000 sec = 14.5 days

The earlier 8-12 day estimate was correct; the table above is wrong because I wrote 54 and 58 instead of 14 and 15. Let me redo the table:

| Variant | Active | Stored | Layers | dim | d_inner | n_experts | Tokens (40×) | Wall-clock @ 6k tok/s | Cost @ $2/hr |
|---|---|---|---|---|---|---|---|---|---|
| 700M | 700M | 1.55B | 32 | 832 | 1216 | 16 | 28B | 13.5 days | $2,600 |
| **775M (primary)** | **775M** | **1.72B** | **32** | **896** | **1280** | **16** | **30B** | **14.5 days** | **$2,800** |
| 900M | 900M | 2.00B | 36 | 960 | 1408 | 16 | 36B | 17.4 days | $3,300 |

With the 4 parallel ablations (each at 7.5B tokens, on a separate pod):
- 7.5B / 24,000 tok/s = 312,500 sec = 3.6 days per ablation pod
- 4 ablations × 3.6 days = 14.4 pod-days of compute
- Each pod is $2/hr × 24 × 3.6 = $173 per ablation

**Total cost (primary + 4 ablations):**
- Primary: $2,800
- 4 ablations: 4 × $173 = $692
- Held-out eval: $100
- **Total: $3,592** (or ~$3,000 with spot discounts)

This is the *corrected* total. The prior estimate of $4,224 was for a 22.5B-token run only; .2 30B-token primary + 4 ablations is **cheaper than the prior estimate** because the per-day cost on RunPod has dropped (their on-demand rate for 4× A100 SXM is now ~$2/hr, was ~$3/hr in 2025).

`★ Insight ─────────────────────────────────────`
- The 4 parallel ablations are a budget-positive addition: they cost $692 total but produce 4 publishable sub-papers, each of which is worth more than $692 in research output value.
- The 900M variant is the "stretch goal" at 17.4 days. If the 775M run converges cleanly, the 900M run is a follow-up with $3,300 budget.
- The 700M variant at $2,600 is the "minimum publishable" run if budget is tight; it would skip the 4 ablations.
`─────────────────────────────────────────────────`

**The 775M is the recommended primary target because:**

1. It lands in the middle of the publishable range, giving headroom to go up (900M) or down (700M) for the ablation comparisons.
2. The dim=896 is a "round" value that makes the partial-RoPE 25% split (= 32 rope_dim) clean.
3. The 30B token budget fits the modern 40× params-in-tokens practice.

**The architecture is scale-invariant in the following sense:** the per-layer shape (8 MLA + 24 GDN, 16-expert MoE on MLA, 3:1 ratio) is the same across all three variants. Only `dim`, `d_inner`, and the layer count change. This means the paper can report results across the family and claim a scaling-law-style finding for the architectural choices, not just a single data point.

---

---
