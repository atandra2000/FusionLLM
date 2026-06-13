<div align="center">

# 🔮 FusionLLM-v1

**Hybrid MLA + GDN + MoE + MTP Pre-training Framework**

[![Python](https://img.shields.io/badge/Python-3.10_–_3.12-3776AB?logo=python&logoColor=fff)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.5-EE4C2C?logo=pytorch&logoColor=fff)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-atandra2000%2FFusionLLM-181717?logo=github)](https://github.com/atandra2000/FusionLLM)

**Single A100 80GB · Pure PyTorch · BF16 · ~415.6M Active Params · ~868.6M Stored**

</div>

---

## ✨ Overview

FusionLLM-v1 is a **single-GPU large language model pre-training framework** that fuses four cutting-edge architectural innovations into a unified 24-layer transformer:

| Component | Layers | Description |
|-----------|--------|-------------|
| **MLA** – Multi-Head Latent Attention | 16 | DeepSeek-style low-rank KV compression with QK-Norm & RoPE |
| **GDN** – Gated Delta Net | 8 | Linear attention via chunked delta-rule recurrence (no softmax) |
| **MoE** – DeepSeek Mixture-of-Experts | 16 (FFN) | 8 routed experts (top-2) + 1 shared expert, aux-loss-free routing |
| **MTP** – Multi-Token Prediction | 2 heads | Auxiliary prediction heads for future-token supervision |

The result: a **415.6M active parameter** model trained on **8.31B Chinchilla-optimal tokens** in ~5.2 days on a single A100, with **868.6M stored parameters** (due to MoE expert proliferation).

---

## 🏗️ Architecture

```
Tokens (B, T)  ──►  Embed (64K × 768)  ──►  24× FusionLLMBlock  ──►  Norm  ──►  Head (tied)
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          │                         │                         │
                    16× MLA Layer             8× GDN Layer            MTP Heads
                    (idx: 0,1,3,4,…)        (idx: 2,5,8,11,…)        (depth=2)
                          │                         │
                    ┌─────┴─────┐             ┌─────┴─────┐
                    │           │             │           │
                 MultiHead    DeepSeek     Gated      Dense
               Latent Attn      MoE       Delta Net   SwiGLU
               (GQA 12:8)   (8+1 experts)  (32 heads)  FFN
                    │           │             │           │
                    └───────────┴─────────────┴───────────┘
                                    │
                              logit softcap (15.0)
```

### Layer Schedule (24 layers)
| Type | Indices | Attention | Feed-Forward | Count |
|------|---------|-----------|--------------|-------|
| **MLA** | 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22 | Multi-Head Latent Attention (GQA 12:8) | DeepSeekMoE (8 routed + 1 shared, top-2) | 16 |
| **GDN** | 2,5,8,11,14,17,20,23 | Gated Delta Net (chunked delta-rule, 32 heads) | Dense SwiGLU FFN (768→2048→768) | 8 |

---

## 📊 Training Recipe

### Hyperparameters

| Setting | Value |
|---------|-------|
| **Optimizer** | NorMuon (lr=0.02, 2D mats) + CautiousAdamW (lr=3e-4) |
| **Scheduler** | WSD (1% warmup, 84% stable, linear decay to 0.1×) |
| **Batch Size** | micro=2, GA=16 → 32 seqs/step (131K tokens) |
| **Precision** | BF16 autocast, SDPA only (no flash, no Triton) |
| **Total Steps** | 63,400 steps (~8.31B tokens) |
| **Sequence Length** | 4096 tokens |
| **Vocabulary** | 64,000 tokens |
| **Checkpoint** | safetensors every 2K steps, max keep 3 |
| **Weight Decay** | 0.1 |

### Key Design Decisions

- **μP Initialization** – Maximal Update Parametrisation for stable training at width
- **Logit Softcap** (15.0) – Prevents logit explosion during early training
- **Tied Embeddings** – Weight-tying between input embed and LM head
- **Aux-Loss-Free Routing** – Biased sigmoid gate with dynamic bias update instead of auxiliary load-balance loss
- **Gradient Checkpointing** – Reduces memory at the cost of ~15% throughput
- **No Triton, No Flash Attention** – Pure PyTorch SDPA for maximum compatibility

---

## 🚀 Quick Start

### Installation

```bash
pip install torch safetensors wandb pyyaml
```

### Build & Inspect the Model

```python
from models.fusionllm import build_fusionllm

config = {
    "vocab_size": 64000,
    "max_seq_len": 4096,
    "dim": 768,
    "n_layers": 24,
    "n_heads": 12,
    "n_kv_groups": 8,
    "q_lora_rank": 192,
    "kv_lora_rank": 96,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "qk_norm": True,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "moe_inter_dim": 2048,
    "inter_dim": 2048,
    "gdn_d_state": 32,
    "gdn_d_conv": 4,
    "gdn_headdim": 32,
    "gdn_d_inner": 1024,
    "gdn_chunk_size": 64,
    "mtp_depth": 2,
    "muP": True,
    "logit_softcap": 15.0,
    "tie_embeddings": True,
}

model = build_fusionllm(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=models --cov=training -v
```

### Pre-training

```python
from training.trainer import Trainer

trainer = Trainer(config)
trainer.train_epoch(data_iter)
# Outputs checkpoints to checkpoints/pretrain/
```

---

## 📁 Project Structure

```
FusionLLM/
├── FINAL_FROZEN_SPEC.md       # Frozen architecture specification (source of truth)
├── IMPLEMENTATION_REPORT.md   # Detailed implementation notes & decisions
├── TEST_PLAN.md               # Test coverage & strategy
├── pyproject.toml             # Project metadata, deps, tooling config
│
├── models/
│   ├── mla.py                 # Multi-Head Latent Attention (GQA + low-rank KV)
│   ├── moe.py                 # DeepSeekMoE with aux-loss-free routing
│   ├── gdn.py                 # Gated Delta Net (linear attention)
│   ├── mtp.py                 # Multi-Token Prediction heads
│   └── fusionllm.py          # Full model assembly (24 layers)
│
├── training/
│   ├── optimizer.py           # NorMuon + CautiousAdamW optimizers
│   ├── scheduler.py          # WSD learning rate schedule
│   ├── checkpoint.py         # safetensors checkpoint save/load
│   ├── validation.py         # Validation loss & perplexity
│   └── trainer.py            # Training loop orchestrator
│
├── tests/
│   ├── test_models.py        # 37 model unit tests
│   └── test_training.py      # 18 training pipeline tests
│
└── archive/                  # Obsolete/legacy code (preserved for reference)
```

---

## 🧠 Component Deep-Dive

### Multi-Head Latent Attention (MLA)
Low-rank KV compression via latent spaces:
- **Q projection**: `768 → 192 (LoRA) → RMSNorm → 12×96 (Q_nope) + 12×32 (Q_pe → RoPE)`
- **KV projection**: `768 → 128 → Split [96 latent, 32 K_pe → RoPE] → RMSNorm → 8×128 (K_nope, V)`
- **Absorption trick**: `Q_nope @ wkv_b_k` for efficient compute
- **Per-layer params**: ~1.16M

### Gated Delta Net (GDN)
Linear attention via delta-rule state update:
- **Input**: `768 → 6×1024 → Split [z, x, b, c, dt, g]`
- **Conv**: Causal depthwise 1D (k=4, groups=1024)
- **Delta rule**: `state = sigmoid(dt·A) * state + outer(k, v)`
- **Read**: `y = C @ state` (+ per-head D skip)
- **Chunked recurrence** (chunk=64), pure PyTorch, FP32 state
- **Per-layer params**: ~8.69M

### DeepSeek MoE
Aux-loss-free biased sigmoid routing:
- **Gate**: Biased sigmoid over 8 experts, top-2 selection
- **Bias update**: Dynamic bias shift (`speed=1e-3`) every 10 steps
- **Experts**: 8 routed (SwiGLU, 768→2048→768) + 1 shared
- **Dispatch**: Pure PyTorch scatter-gather (no Triton)
- **Per-layer params**: ~26.4M (routed) + ~3.15M (shared) = ~29.6M

### Multi-Token Prediction (MTP)
Future-token supervision:
- **Depth 2 heads**: Predict tokens[t+2] (weight=0.10) and tokens[t+3] (weight=0.05)
- **Input**: Concat(main_hidden[t], embed[t+1])
- **Shared transformer block**: MLA + dense SwiGLU FFN
- **Tied output head**: Reuses main embedding weight
- **Additional params**: ~2.46M

---

## 📈 Parameter Breakdown

| Component | Active Params | Stored Params | % of Total |
|-----------|:------------:|:-------------:|:----------:|
| Embedding (tied) | 49.15M | 49.15M | 5.66% |
| MLA (×16) | 18.49M | 18.49M | 2.13% |
| GDN (×8) | 69.51M | 69.51M | 8.00% |
| MoE Routed (×16) | 63.36M | 422.38M | 48.62% |
| MoE Shared (×16) | 50.33M | 50.33M | 5.79% |
| Dense FFN (×8) | 25.17M | 25.17M | 2.90% |
| MTP (depth=2) | 2.46M | 2.46M | 0.28% |
| LM Head (tied) | — | — | — |
| **Total Active** | **415.60M** | **—** | **47.86%** |
| **Total Stored** | **—** | **868.56M** | **100%** |

---

## 🛠️ Tooling

| Tool | Config |
|------|--------|
| **Linter** | Ruff (line-length=120, py310 target) |
| **Formatter** | Ruff (double quotes, space indent) |
| **Testing** | Pytest 8+ with strict markers `gpu` and `slow` |
| **Build** | Hatchling |

```bash
# Lint & format
ruff check models/ training/ tests/
ruff format models/ training/ tests/

# Run specific test categories
pytest tests/ -m "not gpu"   # CPU-only tests
pytest tests/ -m "gpu"       # GPU-required tests
```

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ and PyTorch · Single GPU, Infinite Possibilities</sub>
</div>
