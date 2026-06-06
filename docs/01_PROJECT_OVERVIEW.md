# Project Overview

## High-level Repository Purpose
This repository implements a hybrid language model architecture combining Multi-Head Latent Attention (MLA), Gated Delta Net (GDN/Mamba-2), and Mixture-of-Experts (MoE) components. The model is designed for efficient pre-training and inference with optimized memory usage and computational performance.

## Goals
- Implement a hybrid MLA + GDN architecture with 5:1 ratio (5 attention layers followed by 1 GDN layer)
- Integrate fine-grained DeepSeekMoE with 64 routed experts, 6 activated experts
- Support Multi-Token Prediction (MTP) with depth=3
- Enable μP (μ-transfer) re-initialization for stable training at scale
- Implement comprehensive training pipeline with FSDP2 sharding
- Support curriculum learning for data mixing strategies
- Target 8×A100 SXM 80GB for 150B token training runs

## Current Maturity Level
Based on code analysis, this appears to be a research implementation of a FusionLLM architecture with:
- Phase 2+ features implemented (MLA, MoE enhancements, GDN, MTP, μP, logit softcap)
- Training pipeline with FSDP2 support
- Kernel optimizations for performance
- Configuration-driven architecture selection
- Evaluation harness integration

## Key Innovations
1. **Hybrid Architecture**: Combines MLA (Multi-Head Latent Attention) with GDN (Gated Delta Net) layers in configurable schedules
2. **Enhanced MoE**: Fine-grained routing with group-limited experts and bias-free sigmoid activation
3. **Multi-Token Prediction**: Auxiliary heads for predicting future tokens to improve reasoning
4. **μP Transfer**: Enables stable training hyperparameter transfer from small to large models
5. **Advanced Attention**: MLA with low-rank KV cache, GQA, and optional sliding window
6. **Efficient SSM**: GDN/Mamba-2 layers for constant-time inference
7. **Phase 2.6 Features**: Logit softcap and optional asymmetric rescaling
8. **Comprehensive Kernel Support**: Custom CUDA kernels for operations like fused CE+softcap

## Repository Structure Summary
```
├── kernels/                  # Custom CUDA kernels for performance optimization
├── models/                   # Model architectures (Transformer, MLA, MoE, GDN, etc.)
├── training/                 # Training scripts and utilities
├── configs/                  # Configuration files for training runs
├── data/                     # Data processing utilities
├── evaluation/               # Evaluation harness integration
├── docs/project_context/     # Generated documentation (this directory)
└── requirements.txt          # Python dependencies
```