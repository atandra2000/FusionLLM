<div align="center">

<img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<img src="https://img.shields.io/badge/pytorch-2.7-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.7">
<img src="https://img.shields.io/badge/license-Apache%202.0-3DA639?style=flat-square" alt="Apache 2.0">
<img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="ruff">

# FusionLLM

### Hybrid MLA + GDN + MoE Pre-Training Framework

**State-of-the-art architectural innovations, unified in a production-grade training loop.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [Configuration](#configuration) · [Docs](#documentation)

</div>

---

## Why FusionLLM?

Modern LLMs pick one trick. FusionLLM combines them all — and makes them work together:

| Problem | Solution | Impact |
|---------|----------|--------|
| KV cache blows up at long contexts | **MLA** — low-rank KV compression (5-10× reduction) | Fits longer sequences in the same memory |
| Dense attention is O(n²) | **GDN** — Gated Delta Net, constant-time SSM layers | Every 6th layer runs in linear time |
| Dense FFN wastes compute | **DeepSeekMoE** — 64 routed experts, 6 activated (15.6% active) | 7B total params, only 2.5B active per token |
| Next-token-only training limits reasoning | **MTP** — Multi-Token Prediction, depth=3 | Auxiliary heads predict 3 future tokens |
| Hyperparameters don't transfer across scales | **μP** — μ-transfer re-initialization | Train small, scale up with stable LR |
| Exploding logits destabilize training | **Logit softcap** — `cap · tanh(logits / cap)` | Fused CE+softcap kernel eliminates instability |

**Bottom line**: ~7B total parameters, ~2.5B active, ~4M tokens/sec on 8×A100, 150B tokens in ~10.4 hours.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FusionLLM Backbone                            │
│                                                                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐│
│  │   Block 0  │ │   Block 1  │ │   Block 2  │ │   Block 3  │ │   Block 4  ││
│  │  MLA + MoE │ │  MLA + MoE │ │  MLA + MoE │ │  MLA + MoE │ │  MLA + MoE ││
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘│
│        └───────────────┴───────────────┴───────────────┘              │
│                                   │                                     │
│                         ┌─────────┴─────────┐                           │
│                         │      Block 5       │ ← Every 6th layer        │
│                         │   GDN + Dense FFN  │                           │
│                         └─────────┬─────────┘                           │
│                                   │                                     │
│                            (repeat ×5)                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  MTP Heads (depth=3) · Logit Softcap · Asymmetric Rescale      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │ ParallelEmbed    │  │ RMSNorm + Resid   │  │ Tied LM Head         │ │
│  │ (Vocab-Sharded)  │  │ (Pre-Norm)        │  │ (Shared Weights)      │ │
│  └──────────────────┘  └──────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Schedule**: `5:1` — 5 MLA+MoE blocks, 1 GDN+dense-FFN block, repeated for 30 layers.

### MLA — Multi-Head Latent Attention

Low-rank KV projections compress the cache. Decoupled RoPE preserves positional signal.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `q_lora_rank` | 512 | Query compression rank |
| `kv_lora_rank` | 256 | KV compression rank |
| `n_heads` | 32 | Query heads |
| `n_kv_groups` | 8 | KV groups (4:1 GQA) |
| `qk_nope_head_dim` | 128 | Non-positional dim |
| `qk_rope_head_dim` | 64 | Rotary dim |
| `sliding_window` | 2048 | Local attention window |

### DeepSeekMoE — Fine-Grained Mixture of Experts

Group-limited routing with bias-based load balancing — no auxiliary loss needed.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_routed_experts` | 64 | Total routed experts |
| `n_activated_experts` | 6 | Active per token |
| `n_shared_experts` | 4 | Always-on experts |
| `n_expert_groups` | 8 | Routing groups |
| `n_limited_groups` | 3 | Groups selected per token |
| `moe_inter_dim` | 1536 | Per-expert FFN width |
| `moe_activation` | swiglu | Expert activation |

### GDN — Gated Delta Net

Qwen3-Next style SSM replacing attention every 6th layer. Linear-time inference, constant-time state update.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ssm_type` | gdn | GDN or legacy `mamba2` |
| `gdn_d_state` | 128 | SSM state dimension |
| `gdn_d_conv` | 4 | Temporal conv width |
| `gdn_headdim` | 64 | Per-head dimension |

---

## Quick Start

### Prerequisites

- Python 3.10–3.12
- NVIDIA GPU with CUDA 12.x (A100 80GB recommended)
- PyTorch 2.7+

### Install

```bash
git clone https://github.com/atandra2000/FusionLLM.git
cd FusionLLM

# PyTorch with CUDA 12.8
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install -r requirements.txt

# Or install with extras
pip install -e ".[all]"
```

### Run

```bash
# Minimal smoke test (1 GPU, tiny model)
bash scripts/run_smoke.sh

# Full pre-training (8×A100 SXM 80GB)
bash scripts/run_pretrain_runpod_8xa100.sh

# Single-process with custom config
python training/pretrain.py --config configs/pretrain.yaml
```

---

## Configuration

All settings live in `configs/`. Two entry points:

| File | Use Case |
|------|----------|
| `configs/pretrain.yaml` | Full training — 8×A100, 150B tokens |
| `configs/smoke_pretrain.yaml` | Debug — tiny model, fast iteration |

### Key Parameters

```yaml
model:
  dim: 2048                   # Hidden dimension
  n_layers: 30                # Transformer blocks
  layer_schedule: "5:1"       # MLA:GDN ratio
  n_heads: 32                 # Query heads
  n_kv_groups: 8              # KV groups (GQA)
  vocab_size: 152064          # Qwen2.5 BPE
  max_seq_len: 4096
  mtp_depth: 3                # Multi-Token Prediction depth
  logit_softcap: 15.0         # Caps extreme logits

  # MoE
  n_routed_experts: 64
  n_activated_experts: 6
  n_shared_experts: 4
  moe_inter_dim: 1536

  # GDN
  ssm_type: "gdn"
  gdn_d_state: 128

training:
  micro_batch_size: 2
  gradient_accumulation_steps: 16
  total_steps: 143_000         # ~150B tokens
  lr: 3e-4                     # CautiousAdamW LR
  muon_lr: 0.02                # NorMuon LR (matrix params)
  dtype: bf16
  optimizer: normuon_adamw      # NorMuon + CautiousAdamW
  scheduler: wsd                # Warmup-Stable-Decay
```

### Ablation Shortcuts

| Switch | Set to | Effect |
|--------|-------|--------|
| Disable MoE | `n_routed_experts: 0` | Falls back to dense FFN |
| Pure attention | `layer_schedule: "mha"` | All MLA, no GDN |
| Pure SSM | `layer_schedule: "ssm"` | All GDN, no attention |
| Disable MTP | `mtp_depth: 0` | Standard next-token loss only |
| Disable μP | `muP: false` | Standard initialization |
| Disable softcap | `logit_softcap: 0.0` | Uncapped logits |

---

## Training Pipeline

### Optimizer Strategy

FusionLLM runs **two optimizers** simultaneously:

| Optimizer | Parameters | LR | Key Feature |
|-----------|------------|----|-------------|
| **NorMuon** | Matrix weights (ndim≥2, excl. embed/head) | 0.02 | Newton-Schulz orthogonalization + Adam moments |
| **CautiousAdamW** | Embeddings, norms, biases, LM head | 3e-4 | Sign-masked weight decay (only when grad·param > 0) |

### Learning Rate Schedule

**WSD** (Warmup-Stable-Decay) is the default:

```
LR ▲
   │    ┌────────────────────┐
   │   /                      \
   │  /   stable (84%)         \  decay
   │ /                            \
   └──────────────────────────────────► step
   warmup    stable          decay
   (1%)      (84%)           (15%)
```

### Data Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  Raw Sources                                                     │
│  FineWeb-Edu (60%) · FineMath (15%) · Stack-Edu (15%)           │
│  Cosmopedia (5%) · OpenR1-Math (5%)                              │
└──────────────────────┬───────────────────────────────────────────┘
                       │ tokenize + pack
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Shard Writer → manifest.jsonl + mmap'd .bin shards             │
└──────────────────────┬───────────────────────────────────────────┘
                       │ AsyncShardLoader (micro-prefetch, rank-aware)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Training Loop                                                    │
│  Stage 1: web-heavy (70% FineWeb)                                │
│  Stage 2: code/math-heavy (25% Stack + 25% OpenR1)              │
│  Curriculum switch at configurable step                           │
└──────────────────────────────────────────────────────────────────┘
```

### Distributed Training

FSDP2 `FULL_SHARD` — parameters, gradients, and optimizer states sharded across 8 GPUs:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `fsdp_param_dtype` | bf16 | Reduced memory footprint |
| `fsdp_reduce_dtype` | fp32 | Numerically stable gradients |
| `fsdp_backward_prefetch` | true | Overlap compute and communication |
| `fsdp_forward_prefetch` | false | Saves H2D bandwidth |
| `gradient_checkpointing` | true | ~30-40% activation memory reduction |
| `async_checkpointing` | true | Overlap I/O with training |

**Memory per GPU**: ~4.5 GB static state, ~7.3 GB total estimated on 8×A100 80GB.

---

## Project Structure

```
FusionLLM/
├── models/                     # Neural architectures
│   ├── transformer.py          #   Backbone + ParallelEmbed + TransformerBlock
│   ├── mla.py                  #   Multi-Head Latent Attention
│   ├── moe/                    #   DeepSeekMoE group-limited routing
│   ├── gated_deltanet.py       #   GDN (Qwen3-Next SSM)
│   ├── mamba.py                #   Mamba-2 (legacy SSM)
│   ├── mtp.py                  #   Multi-Token Prediction heads
│   ├── mup.py                  #   μP re-initialization
│   └── rope.py                 #   Rotary Position Embedding
├── training/                   # Training loop & schedulers
│   ├── pretrain.py             #   FSDP2 entry point + ConfigBundle
│   ├── trainer.py              #   Core Pretrainer class
│   ├── normuon.py              #   NorMuon optimizer
│   ├── schedules.py            #   Batch/seq-len ramping
│   ├── wsd.py                  #   Warmup-Stable-Decay scheduler
│   └── loss.py                 #   Loss functions (CE + softcap + MoE + MTP)
├── kernels/                    # Custom CUDA/Triton kernels
│   ├── ce_softcap.py           #   Fused cross-entropy + logit softcap
│   ├── linear_relu2.py         #   Fused linear + ReLU²
│   └── flash_attn.py           #   FlashAttention wrapper
├── ops/                        # Triton kernels
│   └── triton/grouped_gemm.py  #   Grouped GEMM for MoE
├── data/                       # Data pipeline
│   ├── async_loader.py         #   Two-stage async sharded loader
│   ├── curriculum.py           #   Two-stage curriculum switching
│   ├── prepare_data.py         #   Corpus collection + tokenization
│   ├── dedup.py                #   MinHash + exact prefix dedup
│   └── shard_writer.py         #   WebDataset-style sharding
├── eval/                       # Evaluation
│   ├── eval_core.py            #   Perplexity on token loader
│   └── run_lm_eval.py          #   lm-eval-harness integration
├── utils/                      # Utilities
│   ├── distributed.py          #   FSDP2 setup + collectives
│   ├── checkpoint/             #   Atomic/distributed checkpoint I/O
│   └── logging.py              #   W&B + MLflow + CSV
├── configs/                    # YAML configs
├── scripts/                    # Launch scripts
├── tests/                      # Unit + integration tests
└── docs/                       # Full documentation
```

---

## Optional Dependencies

```bash
pip install -e ".[flash]"       # FlashAttention 3
pip install -e ".[kernels]"      # Triton kernels (GDN, CE+softcap, Linear+ReLU²)
pip install -e ".[eval]"         # lm-eval-harness
pip install -e ".[inference]"    # vLLM serving
pip install -e ".[distill]"     # Quantized teachers
pip install -e ".[dev]"          # ruff, pytest, pre-commit
```

---

## Testing

```bash
pytest tests/                                          # All tests
pytest tests/test_mla.py -v                            # Specific module
pytest tests/ -m "not slow and not gpu"                # CPU-only
pytest tests/ -m benchmark --benchmark-only            # Benchmarks
```

---

## Hardware

| Configuration | GPUs | VRAM | Tokens/sec | 150B tokens |
|---------------|------|------|-----------|-------------|
| Smoke test | 1× A100 80GB | 80 GB | — | — |
| Full training | 8× A100 SXM 80GB | 640 GB | ~4.0M | ~10.4 hours |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Project Overview](docs/01_PROJECT_OVERVIEW.md) | Goals, maturity, key innovations |
| [Architecture](docs/02_ARCHITECTURE.md) | Layer stack, MLA, MoE, GDN, MTP details |
| [Training Pipeline](docs/03_TRAINING_PIPELINE.md) | Data, optimizers, schedules, evaluation |
| [Distributed System](docs/04_DISTRIBUTED_SYSTEM.md) | FSDP2, sharding, communication patterns |
| [Configuration Reference](docs/05_CONFIGURATION_REFERENCE.md) | Every config option, defaults, interactions |
| [Memory & Performance](docs/06_MEMORY_AND_PERFORMANCE.md) | VRAM analysis, optimization strategies |
| [Glossary](docs/11_GLOSSARY.md) | Terminology and module reference |

---

## Research References

FusionLLM combines ideas from multiple lines of research:

- **DeepSeek-V2/V3** — MLA, DeepSeekMoE, auxiliary-loss-free bias routing, MTP
- **Qwen3-Next** — Gated Delta Net (GDN) SSM layer
- **Mamba-2** — Selective state space model (legacy option)
- **Keller Jordan** — Muon optimizer (Newton-Schulz orthogonalization)
- **μTransfer** — Stable hyperparameter transfer across scales
- **Jamba / Nemotron-H** — Hybrid attention/SSM schedule patterns
- **FlashAttention** — Fast attention kernel (optional FA3)

---

## Citation

```bibtex
@software{fusionllm2025,
  title   = {FusionLLM: Hybrid MLA + GDN + MoE Pre-Training Framework},
  author  = {FusionLLM Contributors},
  year    = {2025},
  url     = {https://github.com/atandra2000/FusionLLM},
  license = {Apache-2.0}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

**7B parameters. 2.5B active. 10 hours to 150B tokens.**

</div>