<div align="center">

# FusionLLM-v1

**Hybrid Pre-training Framework: Multi-Head Latent Attention + Gated Delta Net + Mixture-of-Experts + Multi-Token Prediction**

[![Python](https://img.shields.io/badge/Python-3.10–3.12-3776AB?logo=python&logoColor=fff)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.5-EE4C2C?logo=pytorch&logoColor=fff)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-3B82F6)](https://github.com/astral-sh/ruff)

**Hardware:** Single A100 80GB &nbsp;·&nbsp; **Precision:** BF16 &nbsp;·&nbsp; **Active Parameters:** 415.6M &nbsp;·&nbsp; **Stored Parameters:** 868.6M &nbsp;·&nbsp; **Training Tokens:** 8.31B

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Training Recipe](#training-recipe)
5. [Benchmarks & Performance](#benchmarks--performance)
6. [Quick Start](#quick-start)
7. [Component Deep-Dive](#component-deep-dive)
8. [Parameter Breakdown](#parameter-breakdown)
9. [Data Pipeline](#data-pipeline)
10. [Project Structure](#project-structure)
11. [Tooling & Testing](#tooling--testing)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

FusionLLM-v1 is a single-GPU large language model pre-training framework that fuses four architectural innovations into a unified 24-layer decoder-only transformer. The framework is designed for efficient pre-training on a single NVIDIA A100 80GB GPU, achieving full Chinchilla-optimal token consumption (~8.31B tokens) in approximately 4–5 days.

The model interleaves **16 Multi-Head Latent Attention (MLA)** layers—inspired by DeepSeek-V2—with **8 Gated Delta Net (GDN)** linear attention layers, providing an efficient hybrid of softmax-based attention and linear-complexity state-space recurrence. Each MLA layer is paired with a **DeepSeek-style Mixture-of-Experts (MoE)** feed-forward block (8 routed experts with top-2 gating, plus 1 shared expert), while each GDN layer uses a dense SwiGLU feed-forward network. **Multi-Token Prediction (MTP)** heads provide auxiliary future-token supervision at depths 2 and 3.

The framework is implemented entirely in **pure PyTorch** (≥2.5) with `torch.compile` kernel fusion and optional Flash Attention 2 integration, requiring no custom CUDA kernels or Triton.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Head Latent Attention** | Low-rank KV compression via latent projections (Q LoRA rank 192, KV rank 96) with QK-Norm and Rotary Position Embeddings |
| **Gated Delta Net** | Linear-complexity attention via chunked delta-rule state update with causal depthwise convolution and FP32-precise recurrence |
| **DeepSeek Mixture-of-Experts** | 8 routed experts (top-2 activation) + 1 shared expert with aux-loss-free biased sigmoid routing and dynamic bias adaptation |
| **Multi-Token Prediction** | Two auxiliary prediction heads (depth 2 with λ=0.10, depth 3 with λ=0.05) using a shared transformer block and tied embedding weights |
| **Dual Optimizer (NorMuon + CautiousAdamW)** | NorMuon optimizer (lr=0.02) for 2D matrix parameters; CautiousAdamW (lr=3e-4) for embeddings, norms, biases, and projections |
| **WSD Scheduler** | Warmup-Stable-Decay schedule: 1% warmup, 84% stable training, 15% linear decay to 0.1× peak learning rate |
| **μP Initialization** | Maximal Update Parameterization for width-scaled initialization and stable training dynamics |
| **Logit Softcap** | Output logit clipping at ±15.0 to prevent early-training divergence |
| **Tied Embeddings** | Weight-tying between input token embedding and output language modeling head |
| **Gradient Checkpointing** | Selective activation checkpointing on MLA layers only, balancing memory and compute |
| **torch.compile** | End-to-end graph capture with `reduce-overhead` mode for 20–40% training throughput improvement |
| **Flash Attention 2** | Optional fused attention kernel for 40–50% faster attention computation on supported CUDA devices |
| **Async Data Loading** | Non-blocking GPU transfers with pinned memory prefetching and multi-worker streaming |
| **Safetensors Checkpointing** | Regular checkpoint save/load with safetensors format (BF16), automatic latest-checkpoint discovery, and retention policy |

---

## Architecture

### Model Topology

```
Tokens (B, T)
    │
    ▼
Embedding (64K × 768)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                 24× FusionLLMBlock                       │
│                                                          │
│   ┌─────────────────────┐   ┌─────────────────────┐     │
│   │  16× MLA Layer      │   │   8× GDN Layer      │     │
│   │  (idx: 0,1,3,4,…)   │   │  (idx: 2,5,8,11,…)  │     │
│   ├─────────────────────┤   ├─────────────────────┤     │
│   │ Multi-Head Latent   │   │ Gated Delta Net     │     │
│   │ Attention (GQA 12:8)│   │ (32 heads, chunk=64)│     │
│   │                     │   │                     │     │
│   │ DeepSeek MoE        │   │ Dense SwiGLU FFN    │     │
│   │ (8+1 experts, top-2)│   │ (768→2048→768)      │     │
│   └─────────────────────┘   └─────────────────────┘     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
RMS LayerNorm
    │
    ▼
LM Head (tied with Embedding) — Logit Softcap (15.0)
    │
    ▼
    │                      ┌─────────────────────────────┐
    ├─────────────────────►│  MTP Heads (depth=2)        │
    │                      │  λ₂=0.10, λ₃=0.05          │
    │                      │  Shared MLA + SwiGLU Block  │
    │                      └─────────────────────────────┘
    │
    ▼
Output Logits (B, T, 64K)
```

### Layer Schedule

The 24 layers follow an alternating schedule designed to maximize the strengths of both attention mechanisms:

| Type | Layer Indices | Attention Mechanism | Feed-Forward Network | Count |
|------|--------------|-------------------|---------------------|:-----:|
| **MLA** | 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22 | Multi-Head Latent Attention (GQA 12 heads, 8 KV groups) | DeepSeekMoE (8 routed + 1 shared expert, top-2) | 16 |
| **GDN** | 2, 5, 8, 11, 14, 17, 20, 23 | Gated Delta Net (32 heads, chunked delta-rule, no softmax) | Dense SwiGLU FFN (768 → 2048 → 768) | 8 |

---

## Training Recipe

### Hyperparameters

| Setting | Value |
|---------|-------|
| **Optimizer** | NorMuon (lr=0.02, β=(0.95, 0.95), 2D matrices) + CautiousAdamW (lr=3e-4, β=(0.9, 0.95), norms/biases/embeddings) |
| **Learning Rate Schedule** | WSD: 1% warmup, 84% stable plateau, 15% linear decay to 0.1× peak |
| **Micro-Batch Size** | 4 sequences |
| **Gradient Accumulation** | 8 steps |
| **Effective Batch Size** | 32 sequences per optimizer step |
| **Tokens per Step** | 131,072 (32 × 4096) |
| **Total Optimizer Steps** | 63,400 |
| **Total Training Tokens** | 8.31B (Chinchilla-optimal for 415.6M active parameters) |
| **Sequence Length** | 4096 tokens |
| **Vocabulary Size** | 64,000 tokens |
| **Precision** | BF16 autocast |
| **Weight Decay** | 0.1 |
| **Gradient Clipping** | 1.0 (global norm) |
| **Logit Softcap** | 15.0 |
| **Checkpointing** | safetensors (BF16), every 2,000 steps, maximum 3 retained |
| **Validation** | Every 5,000 steps on synthetic data (8 batches) |
| **MoE Bias Update** | Dynamic bias shift (speed=1e-3) every 10 steps |
| **Expected Training Time** | ~4–5 days on single A100 80GB |

### Design Rationale

- **μP Initialization:** Maximal Update Parameterization enables stable training at the target width, reducing sensitivity to learning rate choices and improving training dynamics across scales.
- **Dual Optimizer Strategy:** NorMuon applies per-row RMS normalization and sign-masked weight decay for 2D matrix parameters, while CautiousAdamW handles non-matrix parameters with a conservative learning rate. This separation follows findings that different parameter classes benefit from distinct update rules.
- **Selective Gradient Checkpointing:** Only MLA layers are checkpointed, as their low-rank projections produce large intermediate activations. GDN layers and feed-forward networks compute activations on-the-fly, reducing peak memory without full-model checkpointing overhead.
- **Aux-Loss-Free MoE Routing:** Biased sigmoid gating with dynamic bias updates eliminates the need for auxiliary load-balancing losses. The bias shifts gradually encourage expert specialization without interfering with the primary language modeling objective.

---

## Benchmarks & Performance

### Training Throughput

Measured on a single NVIDIA A100 80GB using the integrated benchmark suite:

| Configuration | Measured Throughput | Estimated Training Time (8.31B tokens) |
|--------------|:------------------:|:-------------------------------------:|
| BF16 + torch.compile + Flash Attention 2 | 20,000 – 28,000 tok/s | 3.4 – 4.8 days |
| BF16 + torch.compile (no FA2) | ~15,000 – 18,000 tok/s | ~5.3 – 6.4 days |

### Optimization Impact

| Optimization | Expected Speedup | Source |
|-------------|:---------------:|--------|
| Flash Attention 2 | 40–50% on attention forward/backward | `mla.py` |
| torch.compile (reduce-overhead, fullgraph) | 20–40% overall end-to-end | `trainer.py` |
| BF16 autocast vs FP32 | ~1.8–2.0× arithmetic throughput | `trainer.py` |

### Memory Profile (A100 80GB)

| Component | Memory Estimate |
|-----------|:--------------:|
| Model Weights (BF16, 868.6M params) | ~1.7 GB |
| Optimizer States (NorMuon + CautiousAdamW) | ~5–6 GB |
| Activations (with selective MLA checkpointing) | ~30–40 GB |
| Peak Total (excluding data, cache) | ~50–60 GB |

### Running Benchmarks

```bash
# Standard benchmark (100 steps)
python training/benchmark.py --steps 100

# Extended benchmark for more stable measurement
python training/benchmark.py --steps 500
```

The benchmark constructs the model with `mtp_depth=0`, performs 10 warmup steps for `torch.compile` graph capture, then measures sustained throughput over the specified number of steps using synthetic random data.

---

## Quick Start

### Installation

```bash
# Core dependencies
pip install torch safetensors wandb pyyaml

# Optional: Flash Attention 2 (requires CUDA 12.x, A100 GPU)
pip install flash-attn --no-build-isolation

# Verify Flash Attention installation
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Model Construction

```python
from models.fusionllm import build_fusionllm

config = {
    # Vocabulary & sequence
    "vocab_size": 64000,
    "max_seq_len": 4096,
    # Hidden dimensions
    "dim": 768,
    "n_layers": 24,
    "n_heads": 12,
    "n_kv_groups": 8,
    # MLA low-rank projections
    "q_lora_rank": 192,
    "kv_lora_rank": 96,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "qk_norm": True,
    # MoE configuration
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "moe_inter_dim": 2048,
    # Dense feed-forward
    "inter_dim": 2048,
    # GDN configuration
    "gdn_d_state": 32,
    "gdn_d_conv": 4,
    "gdn_headdim": 32,
    "gdn_d_inner": 1024,
    "gdn_chunk_size": 64,
    # Multi-token prediction
    "mtp_depth": 2,
    # Training infrastructure
    "muP": True,
    "logit_softcap": 15.0,
    "tie_embeddings": True,
}

model = build_fusionllm(config)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Pre-training

```python
from training.trainer import Trainer

trainer = Trainer(config)  # config extended with training hyperparameters
trainer.train_epoch(data_iter)
# Checkpoints written to checkpoints/pretrain/
```

### Configuration Options

The extended config used by `Trainer` supports the following additional keys:

| Option | Default | Description |
|--------|---------|-------------|
| `micro_batch_size` | 4 | Sequences per micro-batch (increase for throughput, reduce for memory) |
| `gradient_accumulation_steps` | 8 | Number of gradient accumulation steps per optimizer update |
| `use_compile` | `True` | Enable `torch.compile` kernel fusion |
| `compile_mode` | `"reduce-overhead"` | Compilation mode: `reduce-overhead` or `max-autotune` |
| `compile_fullgraph` | `True` | Capture entire forward/backward as a single graph |
| `use_checkpoint_per_layer` | `True` | Selective gradient checkpointing on MLA layers |

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Full test suite
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=models --cov=training -v

# CPU-only tests (no GPU required)
pytest tests/ -m "not gpu"

# GPU-required tests
pytest tests/ -m "gpu"
```

---

## Component Deep-Dive

### Multi-Head Latent Attention (MLA)

**File:** `models/mla.py`

MLA replaces the standard multi-head attention with low-rank KV compression via latent projections, as introduced in DeepSeek-V2. This reduces the KV cache size and the computational cost of the attention operation.

**Projection Structure:**
- **Query:** `768 → Q LoRA (192) → RMSNorm → split [12 heads × 96 (nope) + 12 heads × 32 (RoPE)]`
- **Key/Value:** `768 → KV latent (128) → split [96 latent → RMSNorm → 8 heads × 128 (K_nope + V), 32 (K_pe → RoPE)]`

**Key Optimization — Absorption Trick:**
The query non-positional component is multiplied with the absorbed key-value weight matrix (`Q_nope @ wkv_b_k`), eliminating the need to explicitly materialize the full KV representation.

**Flash Attention Integration:**
When available (`flash_attn` package), MLA dispatches to Flash Attention 2 for fused attention compute. Falls back to PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`).

**Per-Layer Parameters:** ~1.16M

### Gated Delta Net (GDN)

**File:** `models/gdn.py`

GDN is a linear-complexity attention mechanism based on a gated delta-rule state update. Unlike softmax attention, GDN maintains a recurrent state that is updated via an outer-product operation, achieving O(T) memory and compute with respect to sequence length.

**Processing Pipeline:**
1. **Input projection:** `768 → 6 × 1024 → split into [z, x, b, c, dt, g]`
2. **Causal depthwise 1D convolution** (kernel size 4, groups 1024, causal padding)
3. **State update (delta rule):** `state ← σ(dt · A) ⊙ state + outer(k, v)`
4. **State read:** `y = C @ state + D ⊙ c` (learned per-head skip connection)
5. **Output gating:** `output = snake(x) ⊙ sigmoid(g)` (element-wise modulation)

**Numerical Precision:**
The recurrent state is maintained in FP32 precision (via explicit `.float()` cast) to mitigate accumulated errors from repeated delta-rule updates. Computation is performed in chunked fashion (chunk size 64) for efficient GPU utilization.

**Per-Layer Parameters:** ~8.69M

### DeepSeek Mixture-of-Experts (MoE)

**File:** `models/moe.py`

The MoE block implements the DeepSeek-style sparse expert architecture with auxiliary-loss-free routing.

**Routing Mechanism:**
- **Gate:** Biased sigmoid over 8 expert logits
- **Selection:** Top-2 experts per token, with scores normalized via softmax
- **Expert dispatch:** Scatter-gather implementation using `torch.index_select` (no Triton dependency)

**Bias Adaptation:**
- Expert biases are updated every `T` steps (default 10) using a small fixed step: `bias ← bias - speed · sign(load - capacity)`
- This shifts routing decisions toward underutilized experts without auxiliary load-balancing losses

**Expert Structure:**
- 8 routed experts, each a SwiGLU MLP: `768 → 2048 (gated) → 768`
- 1 shared expert (always active) with the same SwiGLU structure
- Both use SiLU-gated linear units: `h = (W_gate · x) ⊙ SiLU(W_up · x)` followed by `W_down · h`

**Per-Layer Parameters:** ~29.6M total (routed ~26.4M, shared ~3.15M; stored 422.4M for all routed experts)

### Multi-Token Prediction (MTP)

**File:** `models/mtp.py`

MTP augments the next-token prediction objective with auxiliary future-token prediction heads, following the approach validated in recent large-scale pre-training studies.

**Architecture:**
- **Depth 2 head (λ=0.10):** Predicts token at position `t+2` from `concat(hidden[t], embed[t+1])`
- **Depth 3 head (λ=0.05):** Predicts token at position `t+3` from `concat(hidden[t+1], embed[t+2])`
- Both heads share a single transformer block (MLA + dense SwiGLU FFN)
- Output head reuses the tied main embedding weight

**Loss Integration:**
The MTP losses are weighted and added to the primary next-token prediction loss during training. The small coefficients (0.10, 0.05) ensure the auxiliary objectives do not dominate the primary language modeling objective.

**Additional Parameters:** ~2.46M

---

## Parameter Breakdown

| Component | Active Parameters | Stored Parameters | % of Total |
|-----------|:----------------:|:-----------------:|:----------:|
| Embedding (tied, 64K × 768) | 49,152,000 | 49,152,000 | 5.66% |
| MLA (×16 layers) | 18,489,856 | 18,489,856 | 2.13% |
| GDN (×8 layers) | 69,509,632 | 69,509,632 | 8.00% |
| MoE Routed (×16 layers, 8 experts) | 63,360,000 | 422,380,544 | 48.62% |
| MoE Shared Expert (×16 layers) | 50,331,648 | 50,331,648 | 5.79% |
| Dense SwiGLU FFN (×8 layers) | 25,165,824 | 25,165,824 | 2.90% |
| MTP Heads (depth=2) | 2,459,264 | 2,459,264 | 0.28% |
| LM Head (tied, 0 additional params) | — | — | — |
| **Total Active** | **415,578,624** | **—** | **47.86%** |
| **Total Stored** | **—** | **868,558,848** | **100%** |

The difference between active and stored parameters arises entirely from the MoE routed experts: only 2 of 8 experts per layer are active for any given token, but all 8 expert weight matrices must be stored in memory.

---

## Data Pipeline

The framework includes a complete 6-stage data processing pipeline that transforms raw HuggingFace streaming datasets into memory-mapped `.npy` shards ready for training.

### Pipeline Stages

| Stage | Script | Input → Output | Description |
|:-----:|--------|----------------|-------------|
| 1 | `download_raw.py` | HF streaming datasets → `data/raw/*/shard_*.jsonl.zst` | Streams data from HuggingFace datasets and writes compressed JSONL shards |
| 2 | `preprocess.py` | Raw → `data/clean/*/shard_*.jsonl.zst` | Applies NFKC normalization, PII stripping, length/symbol/URL filtering |
| 3 | `train_tokenizer.py` | Clean sample → `data/tokenizer/tokenizer.model` | Trains 64K byte-level BPE tokenizer using SentencePiece |
| 4 | `tokenize.py` | Clean → `data/tokens/*/tokens_*.bin` | Encodes text to uint16 token sequences with EOS termination |
| 5 | `shard_writer.py` | Tokens → `data/shards/{train,val,test}/shard_*.npy` | Packs tokens into 4096×4096 int32 memory-mapped shards |
| 6 | `streaming_dataloader.py` | Shards → `(tokens, targets)` tensors | Memory-maps shards and yields training batches |

### Dataset Composition

The pipeline builds an 8.31B token training corpus with the following weighted mixture:

| Dataset | Weight | Approx. Tokens | Source |
|---------|:-----:|:--------------:|--------|
| FineWeb-Edu | 55% | 4.57B | `HuggingFaceFW/fineweb-edu` (sample-10BT) |
| FineWeb | 20% | 1.66B | `HuggingFaceFW/fineweb` (sample-10BT) |
| The Stack v2 (Python) | 10% | 0.83B | `bigcode/the-stack-v2-train-full-ids` |
| SlimPajama | 8% | 0.66B | `cerebras/SlimPajama-627B` (dedup) |
| Dolma Wikipedia | 4% | 0.33B | `allenai/dolma` (v1_6-sample), subset=wikipedia |
| Dolma Books | 3% | 0.25B | `allenai/dolma` (v1_6-sample), subset=books |

**Dataset Splits:** Train 97% (8.056B), Validation 1.5% (124.65M), Test 1.5% (124.65M)

### Running the Pipeline

```bash
# Install data pipeline dependencies
pip install ".[data]"

# Execute stages sequentially (each is resumable)
python -m data.scripts.download_raw
python -m data.scripts.preprocess
python -m data.scripts.train_tokenizer
python -m data.scripts.tokenize
python -m data.scripts.shard_writer
```

Each stage is resumable: progress is tracked via JSON state files in `data/state/`, allowing recovery from interruptions without reprocessing completed shards.

---

## Project Structure

```
FusionLLM/
│
├── models/
│   ├── __init__.py          # Public API: model component exports
│   ├── mla.py               # Multi-Head Latent Attention (DeepSeek-V2 style)
│   ├── moe.py               # DeepSeek MoE with aux-loss-free routing
│   ├── gdn.py               # Gated Delta Net (linear attention, delta-rule)
│   ├── mtp.py               # Multi-Token Prediction heads
│   └── fusionllm.py         # FusionLLM model assembly (24 layers, μP init, softcap)
│
├── training/
│   ├── __init__.py          # Public API: training infrastructure exports
│   ├── optimizer.py         # NorMuon + CautiousAdamW + dual optimizer builder
│   ├── scheduler.py         # WSD learning rate schedule (warmup-stable-decay)
│   ├── checkpoint.py        # safetensors save/load, latest discovery, cleanup
│   ├── validation.py        # Validation loss computation, perplexity, shape checks
│   ├── data_loader.py       # Asynchronous data prefetcher (pinned memory)
│   ├── trainer.py           # Main training loop orchestrator
│   └── benchmark.py         # Throughput benchmark (A100-optimized config)
│
├── data/
│   ├── __init__.py
│   ├── common.py            # Shared utilities: I/O, hashing, logging, config loading
│   ├── config/
│   │   ├── mixture.yaml     # Dataset mixture weights (6 sources, 8.31B tokens)
│   │   └── tokenizer.yaml   # SentencePiece BPE tokenizer configuration
│   └── scripts/
│       ├── download_raw.py          # Stage 1: HF streaming → JSONL shards
│       ├── preprocess.py            # Stage 2: cleaning & filtering
│       ├── train_tokenizer.py       # Stage 3: 64K BPE tokenizer training
│       ├── tokenize.py              # Stage 4: text → uint16 token sequences
│       ├── shard_writer.py          # Stage 5: packing → .npy shards
│       └── streaming_dataloader.py  # Stage 6: mmap → (B, T) training tensors
│
├── tests/
│   ├── test_models.py       # 37 model unit tests (param counts, shapes, forward/backward)
│   └── test_training.py     # 18 training pipeline tests (optimizer, scheduler, checkpoint, etc.)
│
├── pyproject.toml           # Project metadata, dependencies, tooling configuration
├── README.md                # This file
└── LICENSE                  # Apache 2.0
```

---

## Tooling & Testing

| Tool | Role | Configuration |
|------|------|---------------|
| **Ruff** | Linter & formatter | Line length 120, Python 3.10+ target, double quotes |
| **Pytest** | Test framework | Pytest 8+ with strict markers: `gpu` and `slow` |
| **Hatchling** | Build system | Wheel packages: `models`, `training`, `data` |

```bash
# Linting
ruff check models/ training/ tests/

# Formatting
ruff format models/ training/ tests/

# Full test suite
pytest tests/ -v --tb=short

# Coverage report
pytest tests/ --cov=models --cov=training -v

# Test categorization
pytest tests/ -m "not gpu"       # CPU-only tests
pytest tests/ -m "gpu"           # GPU-required tests
pytest tests/ -m "slow"          # Long-running tests
```

---

## Citation

If you use FusionLLM-v1 in your research, please cite it as:

```bibtex
@software{fusionllm2025,
  title = {FusionLLM-v1: Hybrid MLA + GDN + MoE + MTP Pre-training Framework},
  author = {FusionLLM Contributors},
  year = {2025},
  url = {https://github.com/atandra2000/FusionLLM},
  description = {Single-GPU large language model pre-training framework fusing Multi-Head Latent Attention, Gated Delta Net, Mixture-of-Experts, and Multi-Token Prediction.}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

<div align="center">
  <sub>Copyright &copy; 2025 FusionLLM Contributors. Licensed under Apache 2.0.</sub>
</div>
