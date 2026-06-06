<div align="center">

# FusionLLM

### Hybrid MLA + GDN + MoE + MTP Pre-Training Framework

**A research-grade, production-ready architecture for efficient large language model pre-training.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://docs.astral.sh/ruff/)

</div>

---

## Overview

FusionLLM combines the best of modern LLM architectural innovations into a single, unified training framework:

| Component | Description |
|-----------|-------------|
| **MLA** | Multi-Head Latent Attention with low-rank KV compression |
| **GDN** | Gated Delta Net (Qwen3-Next style) for constant-time inference |
| **MoE** | Fine-grained DeepSeekMoE with 64 routed experts, 6 activated |
| **MTP** | Multi-Token Prediction (depth=3) for improved reasoning |
| **μP** | μ-transfer re-initialization for stable scaling to large models |

**Target**: ~7B total parameters, ~2.5B active parameters per token.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FusionLLM Backbone                           │
│                                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │   MLA   │ │   MLA   │ │   MLA   │ │   MLA   │ │   MLA   │      │
│  │  + MoE  │ │  + MoE  │ │  + MoE  │ │  + MoE  │ │  + MoE  │      │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│       │           │           │           │           │             │
│       └───────────┴───────────┴───────────┴───────────┘             │
│                              │                                      │
│                      ┌───────┴───────┐                              │
│                      │      GDN      │  ← Every 6th layer          │
│                      │  + Dense FFN  │                              │
│                      └───────┬───────┘                              │
│                              │                                      │
│                        (repeat 5×)                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Multi-Token Prediction                      │    │
│  │            (depth=3, auxiliary prediction heads)             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer Schedule**: `5:1` — 5 MLA layers followed by 1 GDN layer, repeated for 30 total layers.

---

## Key Features

- **Hybrid Architecture** — MLA + GDN in configurable schedules (5:1, 6:1, 8:1)
- **Fine-Grained MoE** — 64 routed experts with group-limited routing and bias-free sigmoid gating
- **Multi-Token Prediction** — Predicts 1, 2, 3 steps ahead for better reasoning
- **μP Transfer** — Stable hyperparameter transfer from small to large models
- **NorMuon + CautiousAdamW** — Dual-optimizer strategy for matrix and non-matrix parameters
- **FSDP2 Sharding** — ZeRO-3 style parameter sharding for multi-GPU training
- **WSD Scheduler** — Warmup-Stable-Decay learning rate schedule
- **Fused Kernels** — Custom CUDA/Triton kernels for CE+softcap, Linear+ReLU², grouped GEMM
- **Curriculum Learning** — Two-stage data mixing (web → code/math)
- **Dual Logging** — W&B + MLflow with CSV persistence

---

## Quick Start

### Prerequisites

- Python 3.10 - 3.12
- NVIDIA GPU with CUDA 12.x (A100 recommended)
- PyTorch 2.7+

### Installation

```bash
# Clone the repository
git clone https://github.com/atandra2000/FusionLLM.git
cd FusionLLM

# Install PyTorch with CUDA 12.8
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install with all extras
pip install -e ".[all]"
```

### Smoke Test (1 GPU)

```bash
bash scripts/run_smoke.sh
```

### Full Pre-Training (8×A100 SXM 80GB)

```bash
bash scripts/run_pretrain_runpod_8xa100.sh
```

### Single-Process Training

```bash
python training/pretrain.py --config configs/pretrain.yaml
```

---

## Configuration

All settings are controlled via YAML files in `configs/`:

| Config | Purpose |
|--------|---------|
| `configs/pretrain.yaml` | Full training profile for 8×A100 SXM 80GB |
| `configs/smoke_pretrain.yaml` | Minimal config for debugging |

### Model Configuration

```yaml
model:
  dim: 2048
  n_layers: 30
  layer_schedule: "5:1"      # 5 MLA + 1 GDN
  n_heads: 32
  n_kv_groups: 8             # GQA: 4 Q heads per KV group
  vocab_size: 152064         # Qwen2.5 BPE tokenizer
  max_seq_len: 4096
  mtp_depth: 3               # Multi-Token Prediction

  # MoE
  n_routed_experts: 64
  n_activated_experts: 6
  n_shared_experts: 4
  moe_inter_dim: 1536

  # GDN
  ssm_type: "gdn"            # or "mamba2" (legacy)
  gdn_d_state: 128
```

### Training Configuration

```yaml
training:
  micro_batch_size: 2
  gradient_accumulation_steps: 16
  total_steps: 143_000       # ~150B tokens
  lr: 3e-4
  muon_lr: 0.02
  dtype: bf16
  optimizer: normuon_adamw
  scheduler: wsd
```

---

## Project Structure

```
FusionLLM/
├── models/                  # Neural network architectures
│   ├── transformer.py       # Main backbone (MLA + GDN schedule)
│   ├── mla.py               # Multi-Head Latent Attention
│   ├── moe/                 # DeepSeekMoE with group-limited routing
│   ├── gated_deltanet.py    # Gated Delta Net (Qwen3-Next)
│   ├── mamba.py             # Mamba-2 SSM (legacy)
│   ├── mtp.py               # Multi-Token Prediction
│   ├── mup.py               # μP re-initialization
│   └── rope.py              # Rotary Position Embedding
├── training/                # Training loop & optimizers
│   ├── pretrain.py          # FSDP2 training entry point
│   ├── trainer.py           # Core training logic
│   ├── normuon.py           # NorMuon optimizer
│   ├── schedules.py         # Batch/seq-len schedulers
│   ├── wsd.py               # Warmup-Stable-Decay scheduler
│   └── loss.py              # Loss functions
├── kernels/                 # Custom CUDA kernels
│   ├── ce_softcap.py        # Fused CE + logit softcap
│   ├── linear_relu2.py      # Fused Linear + ReLU²
│   └── flash_attn.py        # FlashAttention wrapper
├── ops/                     # Triton kernels
│   └── triton/grouped_gemm.py
├── data/                    # Data pipeline
│   ├── async_loader.py      # Async sharded data loader
│   ├── curriculum.py        # Curriculum learning
│   └── prepare_data.py      # Data preparation
├── eval/                    # Evaluation
│   ├── eval_core.py         # Perplexity evaluation
│   └── run_lm_eval.py       # lm-eval-harness integration
├── utils/                   # Utilities
│   ├── distributed.py       # FSDP2 setup & wrapping
│   ├── checkpoint/          # Checkpointing (safetensors/DCP)
│   └── logging.py           # W&B + MLflow logging
├── configs/                 # YAML configurations
├── scripts/                 # Launch scripts
├── tests/                   # Unit & integration tests
└── docs/                    # Documentation
```

---

## Hardware Requirements

| Configuration | GPUs | Memory | Use Case |
|--------------|------|--------|----------|
| Single GPU | 1× A100 80GB | 80 GB | Development / Smoke test |
| Multi-GPU | 8× A100 SXM 80GB | 640 GB | Full pre-training |

**Performance**: ~4.0M tokens/sec on 8×A100 SXM 80GB → 150B tokens ≈ 10.4 hours.

---

## Optional Dependencies

```bash
# Flash Attention 3
pip install -e ".[flash]"

# Triton kernels (GDN, fused CE+softcap, fused Linear+ReLU²)
pip install -e ".[kernels]"

# Evaluation (lm-eval-harness)
pip install -e ".[eval]"

# Inference (vLLM)
pip install -e ".[inference]"

# Quantized teachers for distillation
pip install -e ".[distill]"

# Development tools
pip install -e ".[dev]"
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_mla.py -v

# Run smoke tests only
pytest tests/ -m "not slow and not gpu and not distributed"

# Run benchmarks
pytest tests/ -m benchmark --benchmark-only
```

---

## Documentation

Detailed documentation is available in `docs/`:

| Document | Description |
|----------|-------------|
| [Project Overview](docs/01_PROJECT_OVERVIEW.md) | High-level architecture and goals |
| [Architecture](docs/02_ARCHITECTURE.md) | Detailed model architecture |
| [Training Pipeline](docs/03_TRAINING_PIPELINE.md) | End-to-end training flow |
| [Distributed System](docs/04_DISTRIBUTED_SYSTEM.md) | FSDP2 and distributed training |
| [Configuration Reference](docs/05_CONFIGURATION_REFERENCE.md) | All config options |
| [Memory & Performance](docs/06_MEMORY_AND_PERFORMANCE.md) | Optimization strategies |
| [Research Roadmap](docs/10_RESEARCH_ROADMAP.md) | Future development plans |

---

## Citation

```bibtex
@software{fusionllm2025,
  title  = {FusionLLM: Hybrid MLA + GDN + MoE + MTP Pre-Training},
  author = {FusionLLM Contributors},
  year   = {2025},
  url    = {https://github.com/atandra2000/FusionLLM}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for researchers who value robustness over novelty.**

</div>
