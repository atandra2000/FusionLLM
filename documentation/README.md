# FusionLLM-v1 Documentation

Conceptual reference for the architecture, training stack, and data pipeline of
FusionLLM-v1. These notes capture the *why* and *how* behind the code; the code
itself is kept clean and free of explanatory comments.

> For the authoritative project overview, see the top-level
> [`README.md`](../README.md) and [`AGENTS.md`](../AGENTS.md). This
> `documentation/` folder supplements those with the detailed rationale that
> previously lived inline in the source files.

## Index

| Document | Component | Source file(s) |
|----------|----------|----------------|
| [architecture.md](architecture.md) | 24-layer hybrid topology, μP init, logit softcap, tied embeddings | `models/fusionllm.py` |
| [mla.md](mla.md) | Multi-Head Latent Attention (Q LoRA 192, KV rank 96, decoupled RoPE, absorption trick, FA2) | `models/mla.py` |
| [gdn.md](gdn.md) | Gated Delta Net linear attention (32 heads, chunk 64, FP32 recurrent state, snake/sigmoid gating) | `models/gdn.py` |
| [moe.md](moe.md) | DeepSeekMoE (8 routed top-2 + 1 shared, aux-loss-free biased-sigmoid routing) | `models/moe.py` |
| [mtp.md](mtp.md) | Multi-Token Prediction heads (depth 2 λ=0.10, depth 3 λ=0.05, shared output head) | `models/mtp.py` |
| [fusionllm.md](fusionllm.md) | Top-level `FusionLLMBlock` wiring, forward / `forward_with_hidden`, `generate` | `models/fusionllm.py` |
| [training.md](training.md) | Dual optimizer (NorMuon + CautiousAdamW), WSD scheduler, BF16 + safetensors checkpoints, benchmark | `training/` |
| [data_pipeline.md](data_pipeline.md) | 6-stage pipeline (download → preprocess → 64K BPE → tokenize → 4096×4096 shards → streaming mmap loader), source mix | `data/` |
| [utils.md](utils.md) | Checkpoint, scheduler, validation, benchmark, async data loader notes | `training/` |

## Conventions

- **Raw PyTorch only** — no HuggingFace Trainer, no PyTorch Lightning, no DeepSpeed.
- **No magic numbers** — every architectural constant is defined in code and
  its rationale is recorded here.
- **Separate documentation** — decisions live in this folder, not in code
  comments.