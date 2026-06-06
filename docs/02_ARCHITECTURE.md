# Architecture

## Complete Model Architecture
The model is a hybrid transformer backbone that alternates between Multi-Head Latent Attention (MLA) layers and Gated Delta Net (GDN/Mamba-2) layers based on a configurable schedule. Each layer combines attention/SSM with either a Mixture-of-Experts (MoE) feed-forward network (for MLA layers) or a dense FFN (for GDN layers).

Key components:
- **Embedding System**: Vocab-sharded parallel embedding with tied input/output weights
- **Transformer Blocks**: Alternating MLA and GDN blocks with respective FFNs
- **Final Normalization**: RMSNorm before the language model head
- **LM Head**: Linear layer with optional asymmetric rescaling
- **Auxiliary Heads**: Multi-Token Prediction (MTP) heads for future token prediction

## Layer Stack
Based on the configuration (`layer_schedule: "5:1"` in pretrain.yaml):
- Layers 0-4: MLA blocks (Multi-Head Latent Attention + DeepSeekMoE FFN)
- Layer 5: GDN block (Gated Delta Net + dense SwiGLU FFN)
- This 5:1 pattern repeats for all 30 layers

Schedule parsing logic in `models/transformer.py:parse_schedule()`:
- `"5:1"` means every 6th layer is SSM/GDN (indices 5, 11, 17, 23, 29)
- All other layers use MLA attention

## MLA Details (Multi-Head Latent Attention)
Located in `models/mla.py`:
- **Low-rank KV Cache**: Uses QK LoRA compression with ranks:
  - `q_lora_rank: 512` (query low-rank projection)
  - `kv_lora_rank: 256` (key/value low-rank projection)
- **Decoupled RoPE**: Separate rope dimensions for content and positional:
  - `qk_nope_head_dim: 128` (no positional encoding)
  - `qk_rope_head_dim: 64` (with rotary position encoding)
  - Total head dimension: 192
- **Grouped Query Attention**: `n_heads: 32` total queries, `n_kv_groups: 8` KV heads (4 queries per KV group)
- **Value Dimension**: `v_head_dim: 128`
- **Output Dimension**: `dim: 2048` (matches model dimension)
- **Optional Sliding Window**: `sliding_window: 2048` with `sliding_window_schedule: "5:1"` for local-global interleaving
- **QK Normalization**: `qk_norm: true` for stabilizing training

## MoE Details (DeepSeekMoE)
Located in `models/moe.py`:
- **Expert Count**: `n_routed_experts: 64` (increased from 32 in Phase 2.7)
- **Activated Experts**: `n_activated_experts: 6` tokens routed to 6 experts each
- **Shared Experts**: `n_shared_experts: 4` always-active experts
- **Intermediate Dimension**: `moe_inter_dim: 1536` (per expert FFN size)
- **Router Mechanism**: 
  - Group-limited routing with `n_expert_groups: 8`, `n_limited_groups: 3`
  - `group_topk: 2` (select top 2 experts per group)
  - Bias-update mechanism for load balancing (`bias_update_speed: 1e-3`, `bias_update_every: 10`)
  - Aux-loss-free biased sigmoid routing
- **Capacity**: `expert_capacity_factor: 1.5`, `expert_dropout_prob: 0.1`
- **Activation**: `moe_activation: swiglu` (default) or `relu2` (legacy)
- **Warmup**: `moe_warmup_steps: 2000` steps for expert initialization

## GDN Details (Gated Delta Net)
Located in `models/gated_deltanet.py`:
- **Default Choice**: `ssm_type: "gdn"` (Qwen3-Next style)
- **Legacy Option**: `ssm_type: "mamba2"` (original Mamba-2 selective scan)
- **State Dimension**: `gdn_d_state: 128` (SSM state size)
- **Convolution Width**: `gdn_d_conv: 4` (temporal convolution)
- **Head Dimension**: `gdn_headdim: 64` (per attention head)
- **FFN**: Dense SwiGLU FFN with `inter_dim: 4096` (2× model dimension)

## MTP Details (Multi-Token Prediction)
Located in `training/pretrain.py` and referenced in model config:
- **Depth**: `mtp_depth: 3` (predicts tokens 1, 2, 3 steps ahead)
- **Loss Weight**: `mtp_loss_weight: 0.30` (overall MTP loss contribution)
- **Loss Weight Schedule**: `[0.3, 0.2, 0.1]` (decreasing weight for future tokens)
- **Softcap**: `mtp_softcap: true` with `mtp_softcap_value: 15.0`
- **Implementation**: Auxiliary language model heads sharing the backbone

## Embedding System
- **Parallel Embedding**: `ParallelEmbedding` class for vocab sharding across devices
- **Tied Weights**: `tie_embeddings: true` (input and output embeddings share parameters)
- **Vocab Size**: `vocab_size: 152064` (Qwen2.5 BPE tokenizer)
- **Embedding Dim**: `dim: 2048` (matches model dimension)
- **Initialization**: Normal distribution with std=0.02

## Positional Encoding
- **RoPE (Rotary Position Embedding)**:
  - Base theta: `rope_theta: 10000.0`
  - Factor: `rope_factor: 8.0` (extends context length)
  - Applied to `qk_rope_head_dim: 64` dimensions per head
  - Implemented in `models/rope.py`

## Forward Pass Walkthrough
1. **Input**: Token IDs tensor of shape `[batch_size, seq_len]`
2. **Embedding Lookup**: Vocab-sharded embedding with all-reduce if world_size > 1
3. **Layer Processing**: For each of 30 TransformerBlocks:
   - Pre-norm RMSNorm
   - Either MLA attention or GDN SSM (based on schedule)
   - Residual connection
   - Pre-norm RMSNorm
   - Either MoE FFN (MLA layers) or dense FFN (GDN layers)
   - Residual connection
4. **Final Norm**: RMSNorm before LM head
5. **LM Head**: Linear projection to vocab size (tied with embedding if enabled)
6. **Logit Softcap**: Optional `logit_softcap: 15.0` applied via `cap * tanh(logits / cap)`
7. **Asymmetric Rescale**: Optional per-channel learnable rescaling (disabled by default)
8. **Output**: Logits tensor of shape `[batch_size, seq_len, vocab_size]`

## Parameter Count Estimates
From config `configs/pretrain.yaml`:
- **Model Dimension**: `dim: 2048`
- **Vocab Size**: `vocab_size: 152064`
- **Layers**: `n_layers: 30`
- **Attention Heads**: `n_heads: 32`
- **MLA Compression**:
  - Query projection: `dim → q_lora_rank (512) → qk_nope_head_dim + qk_rope_head_dim (192)`
  - KV projection: `dim → kv_lora_rank (256) → (qk_nope_head_dim + qk_rope_head_dim) * n_kv_groups`
  - Value projection: `dim → kv_lora_rank (256) → v_head_dim * n_kv_groups`
  - Output projection: `(v_head_dim * n_kv_groups) → dim`
- **MoE FFN** (per expert): `dim → moe_inter_dim (1536) → dim` with SwiGLU
- **Dense FFN** (GDN layers): `dim → inter_dim (4096) → dim` with SwiGLU
- **Embedding Tied**: Saves ~2 * vocab_size * dim parameters

Approximate counts (as noted in config comments):
- **Total Parameters**: ~7B
- **Active Parameters**: ~2.5B (due to MoE sparsity: 6/64 routed experts + 4 shared = ~16% active)

Active parameter estimate:
- Routed experts: 6/64 = 9.375%
- Shared experts: 4/64 = 6.25%
- Total MoE activation: ~15.6% of MoE FFN parameters
- Plus attention, GDN, and other dense components

## References to Source Files
- Main model: `models/transformer.py`
- MLA implementation: `models/mla.py`
- MoE implementation: `models/moe.py`
- GDN implementation: `models/gated_deltanet.py`
- Mamba-2 implementation: `models/mamba.py`
- RoPE implementation: `models/rope.py`
- Configuration: `configs/pretrain.yaml`