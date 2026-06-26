# SKILLS.md — FusionLLM

> Skills for the hybrid MLA + GDN + MoE + MTP pre-training framework.
> Pair with `.agents/skills/llm-architecture/SKILL.md` for component-level
> details.

---

## Skill 1: Verify dual-optimizer wiring

Before any training run:

```bash
cd LLM/FusionLLM
grep -n "NorMuon\|CautiousAdamW" training/optimizer.py
grep -n "param_groups" training/trainer.py
```

**Expected:** exactly two param groups. 2D matrices → NorMuon (lr 0.02).
Norms/biases/embeddings → CautiousAdamW (lr 3e-4). Mixing them up silently
degrades training.

## Skill 2: Add a new MLA block type

1. Add the block class in `models/mla.py`.
2. Register it in `FusionLLMBlock`'s `__init__` switch
   (`if self.layer_type == "mla": ...`).
3. Update `models/fusionllm.py:_build_layers()` to schedule the new type.
4. Add a unit test in `tests/test_models.py` matching the 37 existing tests.

**Pitfalls:**
- The absorption trick (`W_Q @ W_UK.T`) must stay on the inference path
  only — training needs material K/V for correct gradients.
- `qk_rope_head_dim=24` is hard-coded in the decoupled RoPE path; changing
  it requires retraining from scratch.

## Skill 3: Add a new GDN chunk size

`chunk_size` lives in `models/gdn.py`. Default 64. Changes:
- Affects numerical stability of the recurrent state.
- Re-tune `sigmoid` vs `snake` gating by ablation (the paper uses snake).

```python
# training/optimizer.py — verify memory before increasing chunk_size:
torch.cuda.reset_peak_memory_stats()
loss = model(batch).backward()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak={peak_gb:.1f} GB")
```

## Skill 4: Add a new data source to the mixture

Edit `data/common.py:MIXTURE`. The 6-stage pipeline will pick it up
automatically on next `download_raw.py` run.

**Pitfalls:**
- Mixture weights must sum to 1.0 ± 1e-6 (the loader asserts).
- All sources must tokenize with the same `64K BPE` tokenizer. Different
  tokenizers cause silent vocab-id collisions.

## Skill 5: Resume from a safetensors checkpoint

```python
from safetensors.torch import load_file
state = load_file("checkpoints/fusionllm_step_50000.safetensors")
model.load_state_dict(state)
```

The trainer's `latest-link` mechanism auto-points to the most recent step.
Use `trainer.train(resume_from="latest")` to continue training.

## Skill 6: Convert between BF16 / FP32 / FP8 (planned)

The model uses BF16 by default. To run in FP32 (debugging):
```python
model = model.to(dtype=torch.float32)
```

To profile FP8 (Hopper only), wrap the linear layers with `transformer_engine`.

## Pitfalls (cross-cutting)
- **Logit softcap ±15.0** is load-bearing — removing it causes early
  divergence on small batches.
- **NorMuon's Newton-Schulz iterations** are 5 steps by default. More
  steps = more compute, no quality gain beyond 5.
- **WSD stable phase** must be ≥80% of total steps; the 84% default is
  tuned for 8B-token runs.

