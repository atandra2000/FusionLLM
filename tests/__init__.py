"""Test package for the fusionllm project.

Layout (one file per architectural unit, per `plan.md:0.3`):

* test_mla.py         — MLA: GQA-on-MLA absorption, sliding-window mask, QK-norm
* test_moe.py         — DeepSeekMoE: group-limited routing, aux-loss-free bias, z-loss
* test_mtp.py         — MTP: depth, shared-head injection
* test_mamba.py       — Mamba-2: parameter shapes, selective-scan reference
* test_rope.py        — RoPE + YaRN (placeholder until Phase 2.1)
* test_muon.py        — Newton-Schulz idempotence (BF16-tolerant)
* test_loader.py      — placeholder for async loader (filled in Phase 1.5)
* test_transformer.py — schedule parser + block dispatcher (the smoke)
* test_distributed.py — FSDP2 helpers: strategy enum, single-GPU no-op
* test_checkpoint.py  — CheckpointManager atomic save/load
* test_logging.py     — TrainerLogger W&B/MLflow shim
* test_eval.py        — eval_core.run_perplexity

Tests are CPU-only by default. GPU-marked tests live behind
`@pytest.mark.gpu` so `make test` never tries to allocate CUDA tensors
on a CPU CI runner.
"""
