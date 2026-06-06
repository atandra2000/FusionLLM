"""Phase 0 eval stub — perplexity on a token loader.

Why a stub?
-----------
Phase 0 exists to land the repo hygiene (Makefile, Dockerfile,
tests, eval hook). The full eval suite (lm-eval-harness wrapper,
HellaSwag / ARC-c / PIQA / WinoGrande / BoolQ / MMLU-lite / GSM8K /
HumanEval, RULER 128 K) lands in Phase 6 alongside the v2
`Pretrainer`.

This module provides exactly one function — `run_perplexity` — plus
a deterministic synthetic loader. The training loop already calls
`run_perplexity` from `Pretrainer._maybe_eval` (also added in Phase
0), so the wiring is in place and the Phase 6 swap is a drop-in.

Function contract
-----------------
``run_perplexity(model, loader, *, device="cpu", max_batches=None)``
returns a dict with three keys:

* ``"loss"``   — mean cross-entropy, as a Python float.
* ``"ppl"``    — ``exp(loss)``, as a Python float.
* ``"n_tokens"`` — total number of target tokens seen.

The function is **rank-0 only** by convention — the caller is
expected to gate with `is_main_process()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F


# ── Synthetic loader (deterministic, used by smoke tests) ──────────────────
def make_synthetic_loader(
    tokenizer: Any = None,
    *,
    n_batches: int = 8,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1024,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(tokens, targets)`` batches of random tokens.

    Deterministic: the same ``seed`` always produces the same
    sequence.  Used by the eval smoke test in `tests/test_eval.py`
    and by `training/pretrain.py` when no real validation set is
    available.
    """
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        # Targets = input shifted left by one (next-token prediction)
        y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        yield x, y


# ── Perplexity ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_perplexity(
    model: Any,
    loader: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None,
    *,
    device: str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute mean cross-entropy and perplexity over ``loader``.

    Args:
        model: a model exposing ``forward(tokens) -> logits`` with
               shape ``(b, s, vocab)``.  Pass ``None`` to use the
               synthetic loader (smoke test path).
        loader: an iterator of ``(tokens, targets)`` tuples.  Pass
                ``None`` for the synthetic loader.
        device: device to move tensors to.
        max_batches: cap on batches (used by smoke tests for speed).

    Returns:
        ``{"loss": float, "ppl": float, "n_tokens": int}``.

    Notes:
        The model is put in ``eval()`` mode for the duration of the
        call and restored to its previous mode afterwards.
    """
    if loader is None:
        loader = make_synthetic_loader()

    if model is not None:
        prev_mode = model.training
        model.eval()
        model = model.to(device)

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for tokens, targets in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        tokens = tokens.to(device)
        targets = targets.to(device)

        if model is not None:
            # Try the project's Transformer signature first; fall back
            # to a bare forward for arbitrary models (used by tests).
            try:
                logits = model(tokens, start_pos=0, use_cache=False)
            except TypeError:
                logits = model(tokens)
        else:
            # Synthetic fallback: random logits for unit testing
            # the eval function in isolation.
            logits = torch.randn(*tokens.shape, 1024, device=device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tok = int((targets != -100).sum().item())
        total_loss += loss.item()
        total_tokens += n_tok
        n_batches += 1

    if model is not None:
        model.train(prev_mode)

    if total_tokens == 0:
        # Defensive: avoid div-by-zero
        return {"loss": float("nan"), "ppl": float("nan"), "n_tokens": 0}

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "ppl": float(torch.tensor(mean_loss).exp().item()),
        "n_tokens": total_tokens,
    }
