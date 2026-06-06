"""Evaluation package for the fusionllm project.

Phase 0 ships a single entry point — `run_perplexity` — that the
training loop calls every `eval_interval` steps.  Phase 6 replaces
this stub with the full lm-eval-harness suite (`run_lm_eval.py`).

Public surface
--------------
* `run_perplexity(model, loader, *, device="cpu", max_batches=None)`
  — standard cross-entropy perplexity on a token loader.  Returns
  ``{"loss": float, "ppl": float, "n_tokens": int}``.
* `make_synthetic_loader(tokenizer=None, *, n_batches=8, batch_size=2,
  seq_len=128, vocab_size=1024, seed=0)`
  — deterministic in-memory loader used by smoke tests.
"""

from .eval_core import make_synthetic_loader, run_perplexity

__all__ = ["run_perplexity", "make_synthetic_loader"]
