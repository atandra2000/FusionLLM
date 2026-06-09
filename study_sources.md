# FusionLLM Interview Prep: End-to-End Study Guide

> **Starting Point Assumption**: You have built GPT-2 from scratch following Andrej Karpathy's video.
> You understand: tokenization, embedding, positional encoding (learned), multi-head attention,
> LayerNorm, GELU MLP, residual connections, cross-entropy loss, autoregressive generation.
>
> This guide explains every concept in FusionLLM by anchoring to what you already know,
> showing exactly what changed, why it changed, and where it lives in this repository.

---

## How to Use This Guide

Each concept follows this template:

```
1. What GPT-2 does (your anchor)
2. What FusionLLM does differently (the gap)
3. Why the change matters (intuition)
4. Where it lives in this repo (file + line references)
5. How to explain it in an interview (script)
6. Study sources to go deeper
7. Common interview questions + model answers
```

---

## Part 1: Attention & Normalization Building Blocks

These are drop-in replacements for GPT-2 components. Same position in the architecture, different math.

---

### 1. RMSNorm — Replacing LayerNorm

#### What GPT-2 Does
LayerNorm: subtract mean, divide by std, scale with gamma, shift with beta.
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

#### What FusionLLM Does
RMSNorm: no mean centering, no beta shift. Just scale by root-mean-square.
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

#### Why The Change
- **Faster**: No mean computation saves ~15% normalization time
- **As effective**: Mean centering doesn't help much in practice for large models
- **Fewer parameters**: No beta (bias) term
- **Standard now**: Used in LLaMA, PaLM, DeepSeek, Qwen, Mistral — basically everything post-2023

#### Where In This Repo
| File | Usage |
|------|-------|
| `models/transformer.py:205-206` | `TransformerBlock.norm1`, `norm2` — pre-norm in every block |
| `models/transformer.py:302` | `Transformer.norm` — final norm before LM head |
| `models/mla.py:95,114` | `q_norm`, `kv_norm` — query/latent normalization |
| `models/mla.py:154-155` | `q_norm_qk`, `k_norm_qk` — QK-norm for training stability |
| `models/mtp.py:102-103,107,109` | Pre-norms inside MTP prediction blocks |

#### Interview Script
> "GPT-2 uses LayerNorm which centers and scales. We use RMSNorm which only scales by the
> root mean square — dropping mean centering and the beta parameter. This gives about 15%
> speedup in normalization with no quality loss. It's the universal standard in modern LLMs.
> We apply it as pre-norm in every transformer block, plus QK-norm on attention for stability."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Original Paper — arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | Understand the theoretical justification for dropping mean centering |
| 2 | [Karpathy's nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | See RMSNorm in ~5 lines — bridge from your GPT-2 knowledge |
| 3 | [Raschka — LLMs from Scratch Ch.4](https://github.com/rasbt/LLMs-from-scratch) | Full implementation walkthrough with LayerNorm → RMSNorm comparison |

#### Common Questions
- **Q: Why not just keep LayerNorm?** A: RMSNorm is 15% faster, no quality regression, and removes a parameter (beta) that doesn't contribute meaningfully at scale.
- **Q: Where exactly do you apply it?** A: Pre-norm on every residual branch (before attention, before FFN), plus QK-norm inside MLA for training stability.

---

### 2. SwiGLU — Replacing GELU MLP

#### What GPT-2 Does
Two linear layers with GELU activation between them:
$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

#### What FusionLLM Does
Three linear layers with Swish (SiLU) gated activation:
$$\text{SwiGLU}(x) = W_3 \cdot (\text{SiLU}(W_1 x) \otimes W_2 x)$$

The key difference: the input is split into two streams — one goes through Swish activation, the other is passed through directly — and they are element-wise multiplied (gated).

#### Why The Change
- **Gating helps**: The element-wise multiply lets the network learn to selectively pass/block information per dimension
- **Swish > GELU for gating**: Swish's non-monotonic shape creates better gradient flow in the gated path
- **3 weight matrices**: Costs more parameters (W1, W2, W3), but the `inter_dim` is typically set to `2/3 * 4 * d_model` to keep total FLOPs similar to GPT-2's MLP
- **Universal**: LLaMA, PaLM, Mistral, Qwen — all use SwiGLU

#### Where In This Repo
| File | Usage |
|------|-------|
| `models/transformer.py:86-112` | `DenseFFN` — standard SwiGLU FFN used in GDN/SSM layers |
| `models/moe/experts.py:16-88` | `Expert` — each MoE expert is a SwiGLU FFN (`inter_dim=1536`) |
| `models/mtp.py:142` | SwiGLU FFN inside each MTP prediction block |

#### Interview Script
> "GPT-2 uses a two-layer MLP with GELU. We replace it with SwiGLU — a three-layer design
> where the input is projected into two paths: one through Swish activation, one passed
> directly, then element-wise multiplied as a gate. This gating mechanism lets the network
> learn which dimensions to pass through. We use inter_dim of 2/3 the normal 4x to keep
> FLOPs constant despite the extra weight matrix."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Shazeer — GLU Variants arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | The original paper — shows SwiGLU outperforms GELU/ReLU variants |
| 2 | [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | See the MLP block — yours has 2 linears, SwiGLU adds a 3rd |
| 3 | [LLaMA paper arXiv:2302.13971](https://arxiv.org/abs/2302.13971) | Section 3.1 — why they chose SwiGLU and the dimension formula |

#### Common Questions
- **Q: Why 3 weight matrices instead of 2?** A: Two matrices create the gate (one activated, one not), and the third projects back. The gate lets each dimension learn to be on/off.
- **Q: Doesn't this double the MLP parameters?** A: We compensate by setting `inter_dim = 2/3 * 4 * d_model` instead of `4 * d_model`, keeping total FLOPs the same.

---

### 3. RoPE — Replacing Learned Positional Embeddings

#### What GPT-2 Does
Learns a separate embedding vector for each position index and adds it to the token embedding:
$$x = \text{token\_embed}(t) + \text{pos\_embed}(pos)$$

#### What FusionLLM Does
Rotary Position Embedding: no learned position vectors. Instead, rotate Q and K vectors in 2D subspaces based on position. Applied *after* the Q/K projection, not added to the input.

$$q_m = \text{Rotate}(q, m\theta), \quad k_n = \text{Rotate}(k, n\theta)$$

where $\theta_i = 10000^{-2i/d}$ and $m, n$ are position indices.

#### Why The Change
- **Length generalization**: Learned embeddings break at positions not seen in training. RoPE encodes relative positions through rotation, so it generalizes to longer sequences.
- **Relative position**: The dot product $q_m^T k_n$ depends on $m-n$ (relative distance), not absolute positions.
- **No extra parameters**: Zero additional parameters (GPT-2 adds `max_seq_len * d_model` parameters).
- **YaRN scaling**: Our implementation supports extending context length by scaling the rotation frequencies.

#### Where In This Repo
| File | Usage |
|------|-------|
| `models/rope.py` (entire file) | `RotaryEmbedding` class with YaRN scaling, complex-valued frequencies |
| `models/mla.py` (throughout) | Applied only to `qk_rope_head_dim=64` dimensions per head (decoupled) |

Key implementation details:
- **Decoupled RoPE**: In MLA, RoPE is applied only to a subset of Q/K dimensions (`qk_rope_head_dim=64`), separate from content dimensions (`qk_nope_head_dim=128`). This resolves the tension between KV compression and position encoding.
- **YaRN scaling**: `rope_factor=8.0` in config for context extension.
- **Complex-valued**: Uses `torch.polar` for frequency computation.

#### Interview Script
> "GPT-2 learns position embeddings that are added to token embeddings. We use RoPE —
> instead of adding position information to the input, we rotate Q and K vectors in 2D
> subspaces based on their position. The rotation angles decrease geometrically across
> dimension pairs. This gives us relative position awareness (the dot product depends on
> position difference) and length generalization (no fixed vocabulary of positions).
> In our MLA implementation, we apply RoPE only to a dedicated 64-dimension slice of Q/K,
> keeping it separate from the content dimensions that get compressed."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [RoFormer paper — arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | The original — understand the rotation matrix derivation |
| 2 | [EleutherAI blog — Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) | Best visual explanation of how 2D rotations create relative positions |
| 3 | [nanoGPT rope.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | Simple implementation to see the core math |
| 4 | [Raschka — LLMs from Scratch Ch.3](https://github.com/rasbt/LLMs-from-scratch) | Step-by-step RoPE implementation |

#### Common Questions
- **Q: How does rotation encode relative position?** A: When you rotate Q by $m\theta$ and K by $n\theta$, their dot product depends on $(m-n)\theta$ — the rotation difference, which is the relative position.
- **Q: Why decoupled RoPE in MLA?** A: MLA compresses KV into a low-rank latent. If RoPE were on all dimensions, compression would destroy positional information. By keeping RoPE on a separate 64-dim slice, we can compress content without losing position.

---

### 4. GQA — Sharing KV Heads

#### What GPT-2 Does
Multi-Head Attention (MHA): every Q head has its own K and V head. With 12 heads, you get 12 Q projections, 12 K projections, 12 V projections.

#### What GPT-2 Does (GQA Variant)
Grouped-Query Attention: multiple Q heads share a single K/V head. In FusionLLM: 32 Q heads, 8 KV groups → each KV head is shared by 4 Q heads.

```
GPT-2 (MHA):    Q1,Q2,...,Q32  →  K1,K2,...,K32  (32 KV heads)
FusionLLM (GQA): Q1,Q2,...,Q32  →  K1,K2,...,K8   (8 KV groups, 4 Q heads each)
```

#### Why The Change
- **KV cache reduction**: During inference, you cache K and V for each token. GQA reduces this cache by `num_q_heads / num_kv_groups` = 4x.
- **Minimal quality loss**: GQA with 8 groups is nearly identical to MHA in quality.
- **Training from MHA checkpoint**: The GQA paper shows you can convert a trained MHA model to GQA by averaging KV heads.

#### Where In This Repo
| File | Usage |
|------|-------|
| `models/mla.py:55-68` | GQA on top of MLA: `n_kv_groups=8`, each KV group serves 4 Q heads |
| `configs/pretrain.yaml` | `n_kv_groups: 8` config setting |

**Critical detail**: In FusionLLM, GQA is layered *on top of* MLA. MLA compresses KV into a latent, then GQA further reduces by sharing KV groups. This gives compound savings: MLA's low-rank compression + GQA's head sharing.

#### Interview Script
> "GPT-2 has every query head paired with its own K and V head — 32 Q, 32 K, 32 V.
> We use GQA where 8 KV groups are shared across 32 Q heads — 4 Q heads per group.
> This cuts the KV cache by 4x at inference. In our architecture, GQA sits on top of MLA,
> so we get compound savings: MLA's low-rank compression plus GQA's head sharing."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [GQA paper — arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | The paper — includes the MHA→GQA conversion technique |
| 2 | [LLaMA-2 paper — arXiv:2307.09288](https://arxiv.org/abs/2307.09288) | Real-world adoption at scale |
| 3 | [HuggingFace LLaMA code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | Clean implementation to compare with your GPT-2 code |

#### Common Questions
- **Q: How do you choose the number of groups?** A: Tradeoff — fewer groups = more KV cache savings but more quality loss. 8 groups is the sweet spot (used by LLaMA-2 70B, Mistral, Qwen2).
- **Q: Can you convert MHA to GQA after training?** A: Yes — the GQA paper shows you can average K heads within each group to initialize GQA from a trained MHA checkpoint, then fine-tune.

---

## Part 2: Architecture-Level Innovations

These are new components that don't exist in GPT-2. They extend the architecture.

---

### 5. MLA (Multi-Head Latent Attention)

#### What GPT-2 Does
Direct attention: project Q, K, V directly from input. Cache full K and V for autoregressive generation.

#### What FusionLLM Does
Compress K and V into a low-rank latent vector, cache only that latent, and reconstruct K and V on-the-fly.

```
GPT-2:    input → W_q → Q,  input → W_k → K,  input → W_v → V  (cache full K, V)
MLA:      input → W_dkv → c_kV (small latent, cache this only)
          c_kV → W_uk → K,  c_kV → W_uv → V  (reconstruct on-the-fly)
```

#### The Core Insight
K and V are redundant — they live in a much lower-dimensional subspace than their raw dimension. By projecting them to a low-rank latent (`kv_lora_rank=256`), caching becomes much cheaper. During attention computation, we reconstruct K and V from the latent.

#### Why The Change
- **KV cache reduction**: Cache `kv_lora_rank=256` per token instead of `2 * n_heads * head_dim = 2 * 8 * 128 = 2048`. That's an 8x reduction.
- **Better than MQA/GQA**: MQA shares 1 KV head (lossy). GQA shares groups (less lossy). MLA compresses into a shared latent (preserves more information).
- **Absorption trick**: We can fold the reconstruction matrices into Q, so during inference the attention computation is just standard dot-product attention.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/mla.py:30-50` | `__init__` | Q LoRA (`q_lora_rank=512`), KV LoRA (`kv_lora_rank=256`) |
| `models/mla.py:55-68` | GQA group config | `n_kv_groups=8`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64` |
| `models/mla.py:90-120` | `forward` | Q projection → split nope/pe → RoPE on pe → KV compression → absorption |
| `models/mla.py:140-160` | Attention | QK-norm → dot product → softmax → weighted V |

#### Interview Script
> "MLA's key idea is that K and V live in a low-rank subspace. Instead of caching full K and V
> vectors — which is 2048 dimensions per token — we project them into a 256-dimensional latent
> vector and cache only that. At attention time, we reconstruct K and V from the latent.
> We combine this with GQA (8 KV groups) for compound savings. The absorption trick lets us
> fold the reconstruction matrices into Q so inference is just standard dot-product attention.
> We also decouple RoPE onto a separate 64-dimensional slice so positional encoding isn't
> destroyed by the compression."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [DeepSeek-V2 paper §2.1 — arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | The original MLA derivation — must read |
| 2 | [DeepSeek-V2 official code](https://github.com/deepseek-ai/DeepSeek-V2) | Reference implementation to compare with ours |
| 3 | [Lilian Weng — Attention mechanisms](https://lilianweng.github.io/posts/2018-06-24-attention/) | Background on attention variants |

#### Common Questions
- **Q: Why not just use MQA or GQA?** A: MQA loses too much expressivity. GQA is better but still shares discrete heads. MLA compresses into a continuous latent, preserving more information per cache byte.
- **Q: What is the absorption trick?** A: We can rewrite `Q @ (W_uk @ c_kV)^T` as `(Q @ W_uk^T) @ c_kV^T` — precomputing `Q @ W_uk^T` means we never need to materialize full K during attention.
- **Q: What's the KV cache math?** A: GPT-2 MHA = `2 * 12 * 64 = 1536` per token. Our MLA = `kv_lora_rank=256` + RoPE dims. That's ~8x reduction.

---

### 6. Sparse MoE & DeepSeekMoE

#### What GPT-2 Does
Dense FFN: every token goes through the same MLP with all parameters active.

$$\text{FFN}(x) = W_2 \cdot \text{SiLU}(W_1 x) \otimes W_2' x$$

#### What FusionLLM Does
Sparse Mixture of Experts: 64 small expert FFNs, a router selects 6 per token, plus 4 shared experts that always run.

```
GPT-2:    token → one FFN (all parameters)
FusionLLM: token → router → select 6 of 64 experts → weighted sum
          + always run 4 shared experts
```

#### Why The Change
- **Scale without compute**: 64 experts × 1536 inter_dim = massive capacity, but only 6 active = constant compute per token
- **Specialization**: Different experts learn different patterns (syntax, semantics, facts, etc.)
- **Shared experts**: Capture universal patterns that all tokens need, reducing redundancy among routed experts
- **Fine-grained**: 64 small experts > 8 large experts — more flexible routing combinations (C(64,6) >> C(8,2))

#### Where In This Repo

**MoE Orchestrator** — `models/moe/moe.py`:
- 64 routed experts + 4 shared experts
- Top-6 routing, capacity factor 1.5, expert dropout 0.1

**Router** — `models/moe/routing.py`:
- `AuxLossFreeGate`: sigmoid-biased scores (not softmax)
- Bias updated by token counts, not auxiliary loss — this is the DeepSeek-V3 innovation
- Group-limited routing: `n_expert_groups=8`, `n_limited_groups=3`, `group_topk=2`

**Experts** — `models/moe/experts.py`:
- Each expert: SwiGLU FFN (`inter_dim=1536`, `dim=2048`)

**Dispatch** — `models/moe/dispatch.py`:
- Three strategies: scatter-gather (default), Triton grouped-GEMM (fast path), all-to-all (distributed)

#### Interview Script
> "Instead of one big FFN for every token, we have 64 small expert FFNs. A router picks
> the top 6 experts for each token, and we also run 4 shared experts that always activate.
> The router uses sigmoid scoring with a bias term that's updated by token counts —
> no auxiliary loss needed. This is DeepSeekMoE's aux-loss-free approach. We use fine-grained
> experts (64 small ones instead of 8 large ones) because the combinatorial flexibility is
> much greater — C(64,6) possible expert combinations vs C(8,2). Group-limited routing
> ensures experts within the same group don't compete."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Switch Transformer — arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | Foundational MoE paper — understand router + auxiliary loss |
| 2 | [DeepSeekMoE — arXiv:2401.06066](https://arxiv.org/abs/2401.06066) | Fine-grained experts + shared experts motivation |
| 3 | [DeepSeek-V2 §3 — arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | Aux-loss-free routing and group-limited routing |
| 4 | [Mixtral paper — arXiv:2401.04088](https://arxiv.org/abs/2401.04088) | Practical MoE at scale — good for comparison |

#### Common Questions
- **Q: What is expert collapse and how do you prevent it?** A: The router learns to send all tokens to 1-2 experts, wasting the rest. DeepSeek-V2 fixes this by using a bias term updated by token counts — if an expert gets too many tokens, its bias decreases, pushing tokens to other experts.
- **Q: Why shared experts?** A: Some patterns (common syntax, frequent tokens) are needed by all inputs. Shared experts capture this, so routed experts can specialize on niche patterns without redundancy.
- **Q: How does group-limited routing work?** A: Experts are divided into 8 groups. Each token can only pick experts from at most 3 groups, with top-2 per group. This prevents routing collapse and improves load balancing.

---

### 7. Mamba-2 (Selective State Space Model)

#### What GPT-2 Does
Full quadratic attention: $O(T^2)$ compute and memory for sequence length T.

#### What FusionLLM Does
Mamba-2 selective SSM: $O(T)$ linear-time sequence modeling. Used as every 6th layer (5:1 MLA-to-GDN ratio).

```
GPT-2:     x → attention(Q,K,V) → output     [O(T²)]
FusionLLM: x → conv1d → selective_scan(h) → output  [O(T)]
```

The recurrence: $h_t = A_t \cdot h_{t-1} + B_t \cdot x_t$, $y_t = C_t \cdot h_t$

Where A, B, C are **input-dependent** (selective) — this is what makes Mamba-2 better than S4.

#### Why The Change
- **Linear complexity**: O(T) vs O(T²) — much cheaper for long sequences
- **Hybrid design**: We don't use Mamba-2 everywhere. MLA handles 5/6 layers (attention for recall/recognition), GDN handles 1/6 layers (SSM for efficient long-range). This hybrid captures the best of both worlds.
- **Complementary strengths**: Attention excels at in-context learning and recall. SSMs excel at smooth, long-range pattern tracking.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/mamba.py:1-120` | Full file | Mamba-2 implementation: `in_proj → conv1d → SiLU → B/C/dt projections → selective_scan → gating → out_proj` |
| `models/gated_deltanet.py:1-150` | Full file | Gated DeltaNet (default SSM type) — the more advanced variant |
| `models/transformer.py:208-213` | Layer schedule | Every 6th layer is GDN (SSM), others are MLA |

#### Interview Script
> "GPT-2 uses full quadratic attention. We use a hybrid: 5/6 layers are MLA attention,
> 1/6 layers are Gated DeltaNet — a state-space model variant. SSMs use a recurrence
> h_t = A_t * h_{t-1} + B_t * x_t, which is O(T) linear time. The selectivity — making
> A, B, C input-dependent — is crucial because it lets the model learn to forget or remember
> based on content. We use this hybrid because attention excels at recall and in-context
> learning, while SSMs are more efficient for long-range pattern tracking."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Mamba paper — arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Original — understand selectivity and hardware-aware scan |
| 2 | [Mamba-2 paper — arXiv:2405.21060](https://arxiv.org/abs/2405.21060) | SSM ↔ linear attention duality — why Mamba-2 is faster |
| 3 | [S4 paper — arXiv:2111.00396](https://arxiv.org/abs/2111.00396) | Foundation — structured state spaces for sequences |
| 4 | [state-spaces/mamba GitHub](https://github.com/state-spaces/mamba) | Official implementation to study |

#### Common Questions
- **Q: Why not use Mamba-2 everywhere instead of attention?** A: Attention has proven capabilities for in-context learning and recall that SSMs struggle with. The hybrid gives us SSM efficiency on most layers while keeping attention where it matters.
- **Q: What does "selective" mean?** A: In S4, the state transition matrices A, B, C are fixed. In Mamba, they're computed from the input — the model learns to selectively remember or forget based on content.

---

### 8. DeltaNet (Gated Delta Net)

#### What GPT-2 Does
Standard attention: $y = \text{softmax}(QK^T / \sqrt{d}) \cdot V$

#### What FusionLLM Does
Delta-rule recurrence with gating: a linear attention variant that uses the delta rule for state updates, with SwiGLU output gating.

```
Standard attention:  y_t = softmax(q_t · K^T / sqrt(d)) · V
DeltaNet:           h_t = h_{t-1} · decay_t + (k_t ⊗ v_t) / (k_t · k_t)
                    y_t = h_t · c_t + v_t · D
                    y_t = y_t · SiLU(z_t) · g_t  (output gating)
```

#### Why The Change
- **Linear complexity**: Like Mamba-2, O(T) instead of O(T²)
- **Better recall than standard linear attention**: The delta rule (from neural network weight updates) allows both positive and negative associations — it can "undo" previous associations, which standard linear attention cannot.
- **Output gating**: The 6-branch projection `(z, x, b, c, dt, g)` gives the model fine-grained control over what information flows through.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/gated_deltanet.py:30-60` | `__init__` | 6-branch input projection: `(z, x, b, c, dt, g)` |
| `models/gated_deltanet.py:60-100` | `forward` | Delta-rule recurrence with D skip and output gating |
| `kernels/delta_rule.py` | Triton kernel | Chunked delta-rule for GPU efficiency |

#### Interview Script
> "DeltaNet is a linear attention variant that uses the delta rule for state updates.
> Standard linear attention accumulates K^T V. DeltaNet instead does h_t = decay * h_{t-1}
> + k_t outer v_t, normalized by k_t dot k_t. This normalization allows the model to
> 'undo' previous associations — the delta rule is borrowed from weight update rules in
> classical neural networks. We use Gated DeltaNet which adds SwiGLU output gating and
> a 6-branch input projection for fine-grained information control."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [DeltaNet paper — arXiv:2106.10153](https://arxiv.org/abs/2106.10153) | Original delta rule for transformers |
| 2 | [Linear Transformers — arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | Foundation: transformers as RNNs via linear attention |
| 3 | [RetNet — arXiv:2307.08621](https://arxiv.org/abs/2307.08621) | Related linear attention variant for comparison |
| 4 | [RWKV — arXiv:2305.13048](https://arxiv.org/abs/2305.13048) | Another linear attention/RNN hybrid |

#### Common Questions
- **Q: How does DeltaNet differ from Mamba-2?** A: Mamba-2 uses a state space model recurrence (A, B, C matrices). DeltaNet uses a delta rule recurrence (outer product updates). Both are O(T), but DeltaNet's delta rule provides better recall through its ability to undo associations.
- **Q: Why the 6-branch projection?** A: Each branch controls a different aspect: output gate (z), value stream (x), B/C keys/values, step size (dt), per-channel gate (g). This gives fine-grained control over what flows through the recurrence.

---

## Part 3: Training Infrastructure

These aren't model architecture changes — they're training system innovations.

---

### 9. FSDP2 (Fully Sharded Data Parallel)

#### What GPT-2 / Karpathy Does
Single-GPU training. All parameters, gradients, and optimizer states live on one GPU.

#### What FusionLLM Does
FSDP2 across 8× A100 GPUs: shards parameters, gradients, and optimizer states across all GPUs.

```
Single GPU:  [parameters | gradients | optimizer states]  ← all on GPU 0
FSDP2:       GPU 0: [shard_0 | shard_0 | shard_0]
             GPU 1: [shard_1 | shard_1 | shard_1]
             ...
             GPU 7: [shard_7 | shard_7 | shard_7]
```

Communication pattern:
- **Forward**: all-gather parameters → compute → discard non-local shards
- **Backward**: all-gather → compute gradients → reduce-scatter gradients → each GPU keeps only its shard

#### Why The Change
- **Memory**: Each GPU holds ~1/8 of parameters instead of all. For a 7B model, this means ~1GB per GPU instead of ~8GB (plus optimizer states).
- **FSDP2 improvements over FSDP1**: Per-parameter sharding (more flexible), better composability with `torch.compile`, backward prefetch tuning.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `utils/distributed.py:257-368` | `wrap_model_fsdp2` | FSDP2 wrapping: `fully_shard` per TransformerBlock |
| `training/trainer.py:107-133` | `setup_distributed` | Distributed initialization and wrapping |
| `configs/pretrain.yaml` | FSDP config | `sharding_strategy: full`, `param_dtype: bf16`, `reduce_dtype: fp32` |

Key design decisions:
- **Per-TransformerBlock wrapping**: Each block is an FSDP unit (not the whole model)
- **Backward prefetch enabled, forward prefetch disabled**: Saves H2D bandwidth
- **bf16 params, fp32 reduction**: Parameters communicated in bf16, gradients reduced in fp32

#### Interview Script
> "FSDP2 shards model parameters, gradients, and optimizer states across GPUs. Before
> forward/backward, each GPU all-gathers the full parameters it needs, computes, then
> discards the non-local shards. We wrap per-TransformerBlock for granularity. Parameters
> are communicated in bf16 but gradients are reduced in fp32 for numerical stability.
> We use backward prefetch to overlap communication with computation."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | Official tutorial — start here |
| 2 | [PyTorch FSDP2 blog](https://pytorch.org/blog/) | FSDP2 announcement — what's new vs FSDP1 |
| 3 | [PyTorch distributed docs](https://pytorch.org/docs/stable/distributed.html) | API reference |
| 4 | [HuggingFace FSDP blog](https://huggingface.co/blog/) | Practical comparison with DeepSpeed |

#### Common Questions
- **Q: FSDP vs DeepSpeed?** A: FSDP2 is PyTorch-native, better composability with `torch.compile`, simpler API. DeepSpeed has more features (ZeRO-3, offloading) but adds a dependency. We chose FSDP2 for cleaner integration.
- **Q: Why per-TransformerBlock wrapping?** A: Granularity. Wrapping the whole model gives one big all-gather. Per-block wrapping allows overlapping communication with computation across layers.

---

### 10. Activation Checkpointing

#### What GPT-2 / Karpathy Does
During forward pass, all intermediate activations are kept in memory for backward pass.

#### What FusionLLM Does
Checkpoint some layers: discard their activations during forward, recompute them during backward.

```
GPT-2:        store all activations → backward reads them
FusionLLM:    checkpoint SSM/GDN layers → recompute during backward
              MLA layers: checkpoint 50% based on checkpoint_mla_ratio
```

#### Why The Change
- **Memory savings**: SSM/GDN layers have large state vectors. Checkpointing them saves significant memory.
- **Selective**: We don't checkpoint everything — MLA layers are checkpointed based on a ratio (`checkpoint_mla_ratio=0.5`), SSM/GDN layers are always checkpointed (they're more expensive to store).
- **~33% compute overhead**: Recomputation costs extra FLOPs, but memory savings enable larger batch sizes.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/transformer.py:208-213` | `_forward` method | `torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)` |
| `models/transformer.py:333-362` | Layer-type-aware policy | SSM/GDN always checkpointed; MLA based on ratio |

#### Interview Script
> "We use layer-type-aware activation checkpointing. SSM/GDN layers are always checkpointed
> because they have large state vectors. MLA layers are checkpointed based on a 50% ratio.
> During forward, activations at checkpoint boundaries are stored. During backward,
> intermediate activations are recomputed on-the-fly. This trades ~33% extra compute for
> significant memory savings, enabling larger batch sizes."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Original paper — arXiv:1604.06174](https://arxiv.org/abs/1604.06174) | The sublinear memory paper |
| 2 | [PyTorch checkpoint tutorial](https://pytorch.org/tutorials/intermediate/gradient_checkpointing.html) | Official implementation guide |
| 3 | [Megatron-LM selective recomputation — arXiv:1910.02054](https://arxiv.org/abs/1910.02054) | Selective vs full — important nuance |

#### Common Questions
- **Q: Why checkpoint SSM layers but not all MLA layers?** A: SSM layers store large state vectors (d_state=128 per head). MLA's activations are smaller. We checkpoint MLA at 50% to balance memory savings vs compute overhead.
- **Q: How do you choose checkpoint boundaries?** A: Per-TransformerBlock. Each block is either fully checkpointed or not, based on layer type and the ratio.

---

### 11. BF16 Mixed Precision

#### What GPT-2 / Karpathy Does
FP32 everywhere: 32-bit floats for parameters, activations, gradients.

#### What FusionLLM Does
BF16 for forward pass, FP32 for gradient reduction:

```
Forward:    autocast to bf16 → compute in bf16
Backward:   gradients computed in bf16 → reduced in fp32
Optimizer:  parameter updates in fp32 → cast to bf16 for storage
```

#### Why The Change
- **2x memory savings**: Parameters and activations use 16 bits instead of 32.
- **BF16 > FP16 for training**: BF16 has 8 exponent bits (same range as FP32), so no loss scaling needed. FP16 only has 5 exponent bits, requiring careful loss scaling.
- **No quality loss**: BF16's lower precision (7 mantissa bits) doesn't matter for training — the noise actually acts as regularization.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `training/trainer.py:180-188` | Forward pass | `torch.amp.autocast("cuda", dtype=torch.bfloat16)` |
| `utils/distributed.py` | FSDP2 config | `param_dtype=bf16`, `reduce_dtype=fp32` |
| `configs/pretrain.yaml:36` | Config | `dtype: bf16` |

#### Interview Script
> "We use BF16 mixed precision. Forward pass runs under autocast in bf16. Gradients are
> reduced in fp32 for numerical stability, then parameters are updated in fp32 and cast
> back to bf16. BF16 is preferred over FP16 because it has 8 exponent bits — same range as
> FP32 — so we don't need loss scaling. The lower mantissa precision acts as implicit
> regularization."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Google BF16 docs](https://cloud.google.com/tpu/docs/bfloat16) | Format specification and rationale |
| 2 | [NVIDIA mixed precision guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/) | Hardware perspective |
| 3 | [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html) | Implementation details |
| 4 | [Tim Dettmers — GPU guide](https://timdettmers.com/) | Format comparison and recommendations |

#### Common Questions
- **Q: Why BF16 instead of FP16?** A: FP16 has 5 exponent bits — small dynamic range. Large gradients overflow, requiring loss scaling. BF16 has 8 exponent bits (same as FP32), so no overflow and no loss scaling needed.
- **Q: Why reduce gradients in fp32?** A: Gradient accumulation across micro-batches can cause precision loss in bf16. FP32 reduction preserves accuracy for the optimizer step.

---

### 12. μP (Maximal Update Parametrization)

#### What GPT-2 Does
Standard Xavier/Kaiming initialization. Learning rate tuned per model size.

#### What FusionLLM Does
μP initialization with scale-invariant hyperparameters:

```python
# Residual stream: std = 1/sqrt(n_layers)  (not 1/sqrt(n_layers))
# Attention/FFN: std = 1/dim
# Embeddings: std = 1/sqrt(dim)
# Gates, biases, A_log, dt_bias: zero-initialized
```

#### Why The Change
- **Hyperparameter transfer**: Tune LR and initialization on a tiny model, transfer directly to the full 7B model — no expensive HP search.
- **Scale invariance**: Optimal hyperparameters don't change with model width.
- **Zero initialization for gates**: Prevents training instability at initialization.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/mup.py` (entire file) | Full muP init | `apply_mup_init()` with residual stream, attention, embed, gate init |
| `models/mup.py:50-80` | `muP_rescale_lr()` | Per-parameter-shape learning rate rescaling |

#### Interview Script
> "μP makes optimal hyperparameters scale-invariant. We initialize residual streams with
> 1/sqrt(n_layers), attention/FFN weights with 1/dim, and embeddings with 1/sqrt(dim).
> All gates and scalar parameters are zero-initialized. This means we can tune the learning
> rate on a small model and transfer it directly to the full model without re-tuning.
> We also implement μP learning rate rescaling for per-parameter-shape LR adjustment."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Tensor Programs V — arXiv:2203.03466](https://arxiv.org/abs/2203.03466) | The μP paper — dense theory, focus on §2-3 |
| 2 | [μP blog](https://parameter-free-transfer-learning.github.io/) | Accessible summary |
| 3 | [nanoGPT μP experiments](https://github.com/karpathy/nanoGPT) | Practical implementation |

#### Common Questions
- **Q: How does μP save time?** A: Instead of running 50 hyperparameter sweeps on the full 7B model, you run them on a 100M model and transfer. That's a 70x reduction in HP search cost.
- **Q: Why zero-initialize gates?** A: At initialization, the gate should pass through the residual stream unchanged. Zero initialization ensures the model starts from the identity function on the residual stream.

---

### 13. Muon / NorMuon Optimizer

#### What GPT-2 / Karpathy Does
AdamW: adaptive learning rates with momentum and weight decay.

#### What FusionLLM Does
Dual optimizer strategy:
- **Matrix parameters** (≥2D): NorMuon (orthogonalized momentum with row-wise RMS normalization)
- **Embeddings, heads, norms, gates**: CautiousAdamW (AdamW with sign-masked weight decay)

```
GPT-2:       AdamW for everything
FusionLLM:   NorMuon (lr=0.02) for weight matrices
             CautiousAdamW (lr=3e-4) for embeddings/head/norms/gates
```

#### Why The Change
- **Muon/NorMuon**: Orthogonalizing the momentum removes redundant gradient components, improving conditioning. Newton-Schulz iteration computes the orthogonal part efficiently.
- **CautiousAdamW**: Weight decay is only applied where `(grad * param).sign() == 1.0` — i.e., where the gradient agrees with the parameter direction. This prevents decay from undoing useful parameter magnitudes.
- **Different LR regimes**: Matrix params benefit from aggressive LR (0.02), while embeddings/norms need conservative LR (3e-4).

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `training/optimization.py:52-108` | `Muon` | Newton-Schulz orthogonalized momentum (5 iterations) |
| `training/normuon.py` | `NorMuon` | Adam + per-row RMS normalization of updates |
| `training/optimization.py:110-150` | `CautiousAdamW` | Sign-masked weight decay |
| `training/optimization.py:150-200` | `build_optimizers` | Dual optimizer: NorMuon for matrices, AdamW for the rest |

#### Interview Script
> "We use a dual optimizer strategy. Matrix parameters use NorMuon — an Adam variant that
> normalizes updates by per-row RMS, giving orthogonalized momentum without explicit SVD.
> Learning rate is 0.02. Everything else — embeddings, LM head, norms, gates — uses
> CautiousAdamW with LR 3e-4. CautiousAdamW only applies weight decay where the gradient
> agrees with the parameter direction, preventing decay from undoing useful magnitudes."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Muon blog — kellerjordan.github.io](https://kellerjordan.github.io/posts/muon/) | Original Muon — Newton-Schulz orthogonalization |
| 2 | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | NanoGPT speedrun with Muon — practical implementation |
| 3 | [Yannic Kilcher — Muon video](https://www.youtube.com/@YannicKilcher) | Visual explanation |

#### Common Questions
- **Q: Why two optimizers?** A: Matrix parameters have different optimal hyperparameters than scalar/embedding parameters. NorMuon's aggressive orthogonalization works for weight matrices but would destroy embedding structure.
- **Q: What is cautious weight decay?** A: Standard AdamW decays all parameters uniformly. CautiousAdamW checks `(grad * param).sign() == 1.0` before decaying — only parameters where gradient agrees with current value get decayed. This prevents useful large-magnitude weights from shrinking.

---

### 14. MTP (Multi-Token Prediction)

#### What GPT-2 Does
Next-token prediction: predict only the immediately following token.

#### What FusionLLM Does
Predict 3 tokens ahead simultaneously with 3 auxiliary prediction heads.

```
GPT-2:       logits = model(x)  →  loss = CE(logits, target)
FusionLLM:   logits_0 = model(x)        →  loss_0 = CE(logits_0, target_0)
             logits_1 = mtp_head_1(h_0)  →  loss_1 = CE(logits_1, target_1)  × 0.3
             logits_2 = mtp_head_2(h_1)  →  loss_2 = CE(logits_2, target_2)  × 0.2
             logits_3 = mtp_head_3(h_2)  →  loss_3 = CE(logits_3, target_3)  × 0.1
             total = loss_0 + 0.3*loss_1 + 0.2*loss_2 + 0.1*loss_3
```

#### Why The Change
- **Richer training signal**: Predicting further ahead forces the model to learn longer-range planning and structure.
- **Speculative decoding at inference**: The auxiliary heads can generate candidate tokens that the main model verifies, speeding up generation.
- **Shared weights**: MTP heads share the main model's embedding and LM head — no extra embedding parameters.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `models/mtp.py:30-60` | `MTPBlock` | Pre-norm transformer block with self-attention + SwiGLU FFN |
| `models/mtp.py:60-100` | `MTPModule` | depth=3, loss schedule [0.3, 0.2, 0.1], softcap CE |
| `models/mtp.py:100-150` | `forward` | Aligned targets, shared embedding/LM head |
| `training/loss.py` | Loss combination | CE + weighted MTP losses + balance loss + z-loss |

#### Interview Script
> "MTP predicts multiple future tokens. Our main model predicts the next token, then three
> auxiliary MTP heads predict tokens 2, 3, and 4 steps ahead. Each MTP head is a small
> pre-norm transformer that takes the hidden state and the embedding of the current token.
> Loss weights decrease: 0.3, 0.2, 0.1. The shared embedding and LM head mean no extra
> parameters. During inference, MTP heads can be used for speculative decoding."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Multi-Token Prediction — arXiv:2404.19737](https://arxiv.org/abs/2404.19737) | Meta's original MTP paper |
| 2 | [DeepSeek-V3 §2.3 — arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | How DeepSeek uses MTP |
| 3 | [Speculative Decoding — arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | Inference acceleration connection |

#### Common Questions
- **Q: How does MTP help at inference?** A: The auxiliary heads can draft candidate tokens. The main model verifies them in one forward pass. If 3 draft tokens are correct, you get 3x speedup.
- **Q: Why decreasing loss weights?** A: Predicting further ahead is harder and noisier. Lower weights prevent the harder tasks from dominating the gradient signal.

---

## Part 4: Custom Kernels

These are GPU-level optimizations that don't change the model math but dramatically improve performance.

---

### 15. Triton — Writing GPU Kernels in Python

#### What GPT-2 / Karpathy Does
Uses PyTorch built-in operations (matmul, softmax, etc.) — no custom kernels.

#### What FusionLLM Does
Custom Triton kernels for fused operations that PyTorch can't optimize automatically:

| Kernel | File | What It Fuses |
|--------|------|---------------|
| Chunked delta-rule | `kernels/delta_rule.py` | Delta-rule state update + output in one kernel |
| CE + softcap | `kernels/ce_softcap.py` | Softcap activation + cross-entropy loss |
| Linear + ReLU² | `kernels/linear_relu2.py` | Matrix multiply + ReLU² activation |
| Grouped GEMM | `ops/triton/grouped_gemm.py` | All expert FFNs in one kernel launch |

#### Why Triton Over CUDA
- **Python-level**: Write kernels in Python, not C++/CUDA
- **Automatic tiling**: Triton handles memory coalescing, shared memory, bank conflicts
- **Autotuning**: Automatically finds optimal block sizes
- **Portability**: Works on NVIDIA GPUs without CUDA toolkit

#### Where In This Repo
| File | Lines | What |
|------|-------|------|
| `kernels/delta_rule.py` | Full | Autotuned chunked delta-rule (BLOCK_HDIM=32/64, CHUNK=32/64) |
| `kernels/ce_softcap.py` | Full | Fused softcap + CE, one program per token |
| `kernels/linear_relu2.py` | Full | Fused matmul + ReLU², 64×64 blocks |
| `ops/triton/grouped_gemm.py` | Full | MoE grouped GEMM, autotuned |

#### Interview Script
> "We use Triton for custom GPU kernels. Triton is a Python DSL for writing efficient
> GPU kernels — the compiler handles tiling, memory coalescing, and shared memory
> automatically. We have four custom kernels: chunked delta-rule for the SSM layers,
> fused CE+softcap for the loss function, fused linear+ReLU² for the FFN, and
> grouped-GEMM for efficient MoE expert computation. All kernels have pure-PyTorch
> fallbacks for portability."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [Triton official tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials) | Start here — 6 progressive tutorials |
| 2 | [Triton paper — arXiv:1905.14255](https://arxiv.org/abs/1905.14255) | Compiler design and rationale |
| 3 | [GPU Mode YouTube](https://www.youtube.com/@gpu_mode) | GPU kernel optimization talks |
| 4 | [Philippe Tillet talk](https://www.youtube.com/@gpu_mode) | Triton creator explaining the design |

#### Common Questions
- **Q: Why not just use PyTorch builtins?** A: PyTorch can't fuse operations across kernel boundaries. A matmul + softcap + CE loss in PyTorch = 3 kernel launches + 2 full tensor writes to HBM. Fused Triton = 1 kernel launch + 1 write.
- **Q: Why Triton instead of raw CUDA?** A: 10x less code, comparable performance. Triton handles the hard parts (memory coalescing, bank conflicts, tiling) automatically.

---

### 16. FlashAttention — IO-Aware Attention

#### What GPT-2 Does
Materializes the full N×N attention matrix in GPU memory (HBM), then computes softmax and V multiplication.

#### What FusionLLM Does
FlashAttention: tiles the attention computation to fit in GPU SRAM (on-chip memory), never materializing the full N×N matrix.

```
GPT-2:            Q, K, V in HBM → compute N×N matrix in HBM → softmax → multiply V
FlashAttention:   Q, K, V in HBM → tile to SRAM → compute softmax+V per tile → write result
```

Memory: O(N²) → O(N). Exact (no approximation).

#### Why The Change
- **Memory**: Standard attention needs O(N²) memory for the attention matrix. FlashAttention uses O(N).
- **Speed**: Reduces HBM reads/writes by tiling — SRAM is 10-100x faster than HBM.
- **Exact**: Not an approximation — mathematically identical to standard attention.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `kernels/flash_attn.py:1-40` | FlashAttention wrapper | Dispatch: FA3 → SDPA fallback |
| `kernels/flash_attn.py:40-80` | Window mask | Gemma 2 style global-local interleaving |

The implementation tries `flash_attn.flash_attn_func` (FlashAttention 3) first, falls back to `F.scaled_dot_product_attention` (PyTorch SDPA).

#### Interview FlashAttention

#### Interview Script
> "FlashAttention is an IO-aware attention algorithm. Instead of materializing the full
> N×N attention matrix in GPU HBM, it tiles the computation into blocks that fit in SRAM.
> Each block computes softmax and V-multiplication on-chip, then writes only the result.
> This reduces memory from O(N²) to O(N) and is 2-4x faster due to reduced HBM traffic.
> It's mathematically exact — same result as standard attention. We use FlashAttention 3
> with SDPA fallback."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [FlashAttention-1 — arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | The IO-awareness insight — must read |
| 2 | [FlashAttention-2 — arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | Work partitioning improvements |
| 3 | [Tri Dao blog — tridao.me](https://tridao.me/) | Author's perspective on the design |
| 4 | [HuggingFace FlashAttention docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2) | Integration guide |

#### Common Questions
- **Q: What is the "online softmax" trick?** A: You can compute softmax one block at a time by maintaining running max and sum. Each block's contribution is corrected by the ratio of old/new normalizers. This enables tiling.
- **Q: Why SRAM vs HBM?** A: A100 has 20MB SRAM but 80GB HBM. SRAM is ~100x faster but much smaller. By tiling into SRAM, we avoid the HBM bottleneck.

---

### 17. Grouped GEMM — Efficient MoE Computation

#### What GPT-2 Does
Single matrix multiply for the FFN: `x @ W.T`.

#### What FusionLLM Does
One grouped GEMM call computes all expert FFNs simultaneously:

```
GPT-2:       y = x @ W.T                    (one GEMM)
FusionLLM:   y = grouped_gemm(x, W, offsets) (one call, 64 experts)
```

Each expert has a different weight matrix and a different number of tokens assigned to it.

#### Why The Change
- **Kernel launch overhead**: 64 separate expert GEMMs = 64 kernel launches. Grouped GEMM = 1 launch.
- **Better GPU utilization**: Parallelizes across experts and output dimensions in a single kernel.
- **Autotuned**: Block sizes are automatically selected per configuration.

#### Where In This Repo
| File | Key Lines | What |
|------|-----------|------|
| `ops/triton/grouped_gemm.py:1-80` | Full kernel | Autotuned Triton grouped-GEMM |
| `models/moe/dispatch.py` | Expert dispatch | Chooses grouped-GEMM path when conditions met |

#### Interview Script
> "Instead of launching 64 separate expert GEMMs, we use a single grouped-GEMM kernel.
> The input is partitioned into groups by expert assignment, and the kernel processes all
> experts in one launch. The grid is (n_experts, output_dim / BLOCK_N), parallelizing over
> experts and output dimensions. This eliminates kernel launch overhead and improves GPU
> utilization. The block sizes are autotuned."

#### Study Sources

| # | Source | Why Read It |
|---|--------|-------------|
| 1 | [cuBLAS grouped GEMM docs](https://docs.nvidia.com/cublas/) | NVIDIA's reference implementation |
| 2 | [CUTLASS grouped GEMM examples](https://github.com/NVIDIA/cutlass/tree/main/examples) | CUDA-level implementation |
| 3 | [SGLang grouped GEMM](https://github.com/sgl-project/sglang) | Production implementation |
| 4 | [vLLM MoE kernels](https://github.com/vllm-project/vllm) | Another reference |

#### Common Questions
- **Q: How do you handle irregular group sizes?** A: Each expert may have a different number of tokens. The kernel uses an offset array to track where each expert's tokens start/end in the batched input.
- **Q: Why not use `torch.bmm`?** A: `torch.bmm` requires all groups to have the same size. Expert assignments are irregular. Grouped GEMM handles variable-size groups natively.

---

## Part 5: Putting It All Together — The Architecture

### FusionLLM Architecture Diagram

```
Input: Token IDs [B, T]
  │
  ├── ParallelEmbedding (vocab-sharded across GPUs)
  │
  ├── Layer 0-4: MLA Block (×5)
  │   ├── RMSNorm → MLA Attention (Q LoRA → RoPE → KV latent → FlashAttn) → Residual
  │   └── RMSNorm → DeepSeekMoE FFN (64 experts, top-6 + 4 shared) → Residual
  │
  ├── Layer 5: GDN Block (×1)
  │   ├── RMSNorm → Gated DeltaNet (6-branch → conv1d → delta-rule → gating) → Residual
  │   └── RMSNorm → Dense SwiGLU FFN → Residual
  │
  ├── Layer 6-10: MLA Block (×5)    ← pattern repeats
  │   ...
  │
  ├── Layer 29: GDN Block (×1)
  │
  ├── Final RMSNorm
  │
  ├── LM Head (tied with embedding)
  │
  └── Logit Softcap (15.0 * tanh(logits / 15.0))
        │
        ├── Main CE Loss (next-token)
        ├── MTP Loss ×3 (tokens 2,3,4 ahead, weights 0.3/0.2/0.1)
        ├── MoE Balance Loss
        └── Z-Loss (numerical stability)
```

### How Every Concept Maps to GPT-2

| GPT-2 Component | FusionLLM Replacement | Key Difference |
|----------------|----------------------|----------------|
| LayerNorm | RMSNorm | No mean centering, no beta |
| Learned position embed | RoPE | Rotation, no params, relative position |
| GELU MLP | SwiGLU FFN | Gated activation, 3 weight matrices |
| MHA (all heads) | MLA + GQA | Low-rank KV compression + head sharing |
| Single FFN | DeepSeekMoE (64 experts) | Sparse routing, capacity scaling |
| — | GDN (SSM layer) | Linear-time sequence modeling |
| FP32 training | BF16 mixed precision | 2x memory savings, no loss scaling |
| AdamW | NorMuon + CautiousAdamW | Orthogonalized momentum + cautious decay |
| Next-token only | MTP (3 extra tokens) | Richer training signal |
| PyTorch builtins | Triton fused kernels | 1 kernel vs 3 |
| In-memory attention | FlashAttention | O(N) vs O(N²) memory |
| Single expert GEMM | Grouped GEMM | 1 launch vs 64 |

---

## Study Plan: 4-Week Interview Prep

### Week 1: Attention & Normalization (Foundation)
**Goal**: Understand every drop-in replacement for GPT-2 components.

| Day | Topic | Read | Implement |
|-----|-------|------|-----------|
| 1 | RMSNorm | Original paper + nanoGPT code | Write RMSNorm from scratch, compare to LayerNorm |
| 2 | SwiGLU | Shazeer paper + LLaMA §3.1 | Modify GPT-2 MLP to SwiGLU, verify shapes |
| 3 | RoPE | RoFormer paper + EleutherAI blog | Implement RoPE, visualize rotation matrices |
| 4 | GQA | GQA paper + HuggingFace LLaMA code | Modify attention to GQA, measure KV cache reduction |
| 5 | MLA | DeepSeek-V2 §2.1 | Implement MLA, compare KV cache sizes |
| 6-7 | Review + mock interview | All of the above | Explain each concept without notes |

### Week 2: Architecture & MoE
**Goal**: Understand the full model architecture and routing.

| Day | Topic | Read | Implement |
|-----|-------|------|-----------|
| 1 | Sparse MoE | Switch Transformer + ST-MoE | Implement router + top-k dispatch |
| 2 | DeepSeekMoE | DeepSeekMoE paper + V2 §3 | Add shared experts, aux-loss-free routing |
| 3 | Mamba-2 | Mamba + Mamba-2 papers | Implement selective scan (PyTorch reference) |
| 4 | DeltaNet | DeltaNet + Linear Transformers papers | Implement delta-rule recurrence |
| 5 | Architecture | Read `transformer.py` end-to-end | Trace the full forward pass |
| 6-7 | Review + mock interview | All of the above | Draw architecture from memory |

### Week 3: Training Infrastructure
**Goal**: Understand distributed training and optimization.

| Day | Topic | Read | Implement |
|-----|-------|------|-----------|
| 1 | FSDP2 | PyTorch FSDP tutorial + blog | Wrap a small model with FSDP2 |
| 2 | Activation Checkpointing | Original paper + PyTorch tutorial | Apply checkpointing, measure memory savings |
| 3 | BF16 | Google BF16 docs + PyTorch AMP | Train with bf16, compare to fp32 |
| 4 | μP | Tensor Programs V + nanoGPT μP | Implement μP init, compare to Xavier |
| 5 | Muon | Muon blog + modded-nanogpt | Implement Muon optimizer |
| 6-7 | Review + mock interview | All of the above | Explain communication patterns |

### Week 4: Kernels & Integration
**Goal**: Understand the GPU-level optimizations and tie everything together.

| Day | Topic | Read | Implement |
|-----|-------|------|-----------|
| 1 | Triton basics | Triton tutorials 1-3 | Write a fused matmul + bias kernel |
| 2 | FlashAttention | FA1 + FA2 papers | Implement tiled attention (simplified) |
| 3 | Grouped GEMM | CUTLASS examples + Triton tutorial | Implement grouped GEMM for MoE |
| 4 | MTP | Meta MTP paper + DeepSeek-V3 §2.3 | Implement MTP loss computation |
| 5 | End-to-end | Read full codebase | Trace forward + backward pass |
| 6-7 | Mock interviews | All concepts | 30-min mock interview with all topics |

---

## Mock Interview: 20 Questions

### Architecture Questions
1. "Walk me through the forward pass of your model." → Trace from embedding to loss.
2. "Why use MLA instead of GQA?" → Low-rank compression vs discrete head sharing.
3. "How does your MoE routing work?" → Aux-loss-free sigmoid + bias + group-limited.
4. "Why a hybrid 5:1 MLA-to-GDN ratio?" → Attention for recall, SSM for efficiency.
5. "Explain the absorption trick in MLA." → Fold reconstruction into Q.

### Implementation Questions
6. "How do you handle the KV cache with MLA?" → Cache latent, reconstruct K/V on-the-fly.
7. "Why use Triton instead of PyTorch builtins?" → Fusion across kernel boundaries.
8. "How does activation checkpointing interact with FSDP2?" → Per-block checkpointing, FSDP wrapping.
9. "Why NorMuon over AdamW for matrix parameters?" → Orthogonalized momentum, better conditioning.
10. "How does grouped-GEMM handle irregular expert sizes?" → Offset array, variable-size groups.

### System Design Questions
11. "How would you scale this to 64 GPUs?" → Hierarchical FSDP, expert parallelism for MoE.
12. "What's the communication bottleneck?" → All-gather in FSDP, all-to-all in MoE.
13. "How do you handle loss spikes?" → EMA detection, z-score analysis, emergency checkpointing.
14. "Why BF16 for params but FP32 for gradient reduction?" → Memory savings vs numerical stability.
15. "How does MTP help inference?" → Speculative decoding with auxiliary heads.

### Theory Questions
16. "Why does RMSNorm work without mean centering?" → Mean centering doesn't contribute to gradient signal at scale.
17. "What is the dual of SSMs?" → Linear attention (Mamba-2's key insight).
18. "How does μP enable zero-shot HP transfer?" → Scale-invariant update rules.
19. "Why does the delta rule improve recall over standard linear attention?" → Can undo previous associations.
20. "Explain the IO complexity of FlashAttention." → SRAM tiling reduces HBM traffic from O(N²) to O(N).

---

## All Study Sources (Consolidated)

### Papers (Must Read)
| # | Paper | Topic |
|---|-------|-------|
| 1 | [RMSNorm — arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | Normalization |
| 2 | [GLU Variants — arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | SwiGLU |
| 3 | [RoFormer — arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | RoPE |
| 4 | [GQA — arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | GQA |
| 5 | [DeepSeek-V2 — arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | MLA + MoE |
| 6 | [DeepSeekMoE — arXiv:2401.06066](https://arxiv.org/abs/2401.06066) | Fine-grained MoE |
| 7 | [Switch Transformer — arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | MoE foundations |
| 8 | [Mamba — arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Selective SSM |
| 9 | [Mamba-2 — arXiv:2405.21060](https://arxiv.org/abs/2405.21060) | SSM-attention duality |
| 10 | [DeltaNet — arXiv:2106.10153](https://arxiv.org/abs/2106.10153) | Delta rule |
| 11 | [Tensor Programs V — arXiv:2203.03466](https://arxiv.org/abs/2203.03466) | μP |
| 12 | [Multi-Token Prediction — arXiv:2404.19737](https://arxiv.org/abs/2404.19737) | MTP |
| 13 | [FlashAttention-1 — arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | IO-aware attention |
| 14 | [FlashAttention-2 — arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | FA improvements |
| 15 | [Triton — arXiv:1905.14255](https://arxiv.org/abs/1905.14255) | GPU kernel compiler |

### Codebases (Study These)
| # | Codebase | What to Study |
|---|----------|---------------|
| 1 | [This repo — FusionLLM](.) | Everything — your primary reference |
| 2 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | GPT-2 baseline — your starting point |
| 3 | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba-2 reference implementation |
| 4 | [deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) | MLA and MoE reference |
| 5 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | FlashAttention implementation |
| 6 | [triton-lang/triton](https://github.com/triton-lang/triton) | Triton tutorials and examples |
| 7 | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Muon optimizer implementation |
| 8 | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Step-by-step implementation guide |

### Video Resources
| # | Channel | Best For |
|---|---------|----------|
| 1 | [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | GPT-2 fundamentals, nanoGPT |
| 2 | [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper explanations for most topics |
| 3 | [The AI Epiphany](https://www.youtube.com/@theaiepiphany) | Architecture deep dives |
| 4 | [GPU Mode](https://www.youtube.com/@gpu_mode) | Triton, FlashAttention, kernels |

### Blogs & Tutorials
| # | Resource | Covers |
|---|----------|--------|
| 1 | [jalammar.github.io](https://jalammar.github.io/) | Visual transformer explanations |
| 2 | [lilianweng.github.io](https://lilianweng.github.io/) | Attention, MoE, memory optimization |
| 3 | [blog.eleuther.ai](https://blog.eleuther.ai/) | RoPE, training techniques |
| 4 | [tridao.me](https://tridao.me/) | FlashAttention author's blog |
| 5 | [kellerjordan.github.io](https://kellerjordan.github.io/) | Muon optimizer |
| 6 | [pytorch.org/tutorials](https://pytorch.org/tutorials/) | FSDP, AMP, checkpointing |
| 7 | [huggingface.co/blog](https://huggingface.co/blog/) | Training techniques, MoE |

---

*Last updated: 2026-06-07*
*Repository: FusionLLM — Hybrid MLA + DeepSeekMoE + GDN Architecture*
*Target: Interview-ready understanding of all 20 concepts as implemented in this codebase*
