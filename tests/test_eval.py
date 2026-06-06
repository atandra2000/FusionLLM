"""Unit tests for `eval/eval_core.py`.

Phase 0 scope (per `plan.md:0.3`):
  * `make_synthetic_loader` — deterministic, correct shape.
  * `run_perplexity` — no-model smoke (uses the synthetic logits
    path) and with a real (randomly-initialised) model. Returns
    ``loss``, ``ppl``, ``n_tokens`` with the expected types.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from eval.eval_core import make_synthetic_loader, run_perplexity


# ── make_synthetic_loader ──────────────────────────────────────────────────
class TestSyntheticLoader:
    def test_deterministic(self):
        a = list(make_synthetic_loader(n_batches=2, batch_size=2, seq_len=16, seed=42))
        b = list(make_synthetic_loader(n_batches=2, batch_size=2, seq_len=16, seed=42))
        for (xa, ya), (xb, yb) in zip(a, b):
            assert torch.equal(xa, xb)
            assert torch.equal(ya, yb)

    def test_shapes(self):
        x, y = next(make_synthetic_loader(n_batches=1, batch_size=3, seq_len=10, vocab_size=64))
        assert x.shape == (3, 10)
        assert y.shape == (3, 10)

    def test_target_is_left_shift(self):
        x, y = next(make_synthetic_loader(n_batches=1, batch_size=1, seq_len=8))
        # The last column of y wraps to x[:, 0]
        assert torch.equal(y[:, :-1], x[:, 1:])
        assert torch.equal(y[:, -1], x[:, 0])

    def test_n_batches(self):
        batches = list(make_synthetic_loader(n_batches=5))
        assert len(batches) == 5


# ── run_perplexity (no-model smoke) ───────────────────────────────────────
class TestRunPerplexityNoModel:
    def test_returns_expected_keys(self):
        result = run_perplexity(model=None, max_batches=2)
        assert set(result.keys()) == {"loss", "ppl", "n_tokens"}
        for k in ("loss", "ppl", "n_tokens"):
            assert isinstance(result[k], float) or isinstance(result[k], int)

    def test_loss_is_positive(self):
        result = run_perplexity(model=None, max_batches=2)
        # random logits → cross-entropy is positive
        assert result["loss"] > 0.0

    def test_ppl_equals_exp_loss(self):
        import math

        result = run_perplexity(model=None, max_batches=2)
        assert result["ppl"] == pytest.approx(math.exp(result["loss"]), rel=1e-5)

    def test_n_tokens_scales_with_batches(self):
        a = run_perplexity(model=None, max_batches=1)
        b = run_perplexity(model=None, max_batches=2)
        # Same batch_size and seq_len per call → tokens scale linearly
        assert b["n_tokens"] == 2 * a["n_tokens"]


# ── run_perplexity (with a real, tiny model) ──────────────────────────────
class TestRunPerplexityWithModel:
    def test_with_tiny_linear_model(self):
        # The simplest possible model: a single Linear layer.
        # logits = W @ x + b  (vocab_size = output dim)
        class TinyLM(nn.Module):
            def __init__(self, dim: int, vocab_size: int):
                super().__init__()
                self.proj = nn.Linear(dim, vocab_size, bias=False)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                # tokens are used as one-hot inputs (toy)
                x = F.one_hot(tokens, num_classes=dim).float()
                return self.proj(x)

        import torch.nn.functional as F

        dim = 32
        vocab = 64
        m = TinyLM(dim=dim, vocab_size=vocab)

        # Custom loader: yield tokens in [0, dim) so one-hot is well-defined
        def loader():
            g = torch.Generator().manual_seed(0)
            for _ in range(2):
                x = torch.randint(0, dim, (2, 8), generator=g)
                y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
                yield x, y

        result = run_perplexity(m, loader(), device="cpu", max_batches=2)
        assert result["loss"] > 0.0
        assert result["ppl"] > 1.0
        assert result["n_tokens"] == 2 * 8 * 2

    def test_model_is_restored_to_train_mode(self):

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                # Keep logits in a known vocab so the synthetic loader
                # (which emits tokens in [0, 1024)) can stay valid.
                self.proj = nn.Linear(8, 2048, bias=False)

            def forward(self, tokens):
                # Map tokens into the input dim; produce logits.
                return self.proj(torch.zeros(*tokens.shape, 8))

        m = TinyLM()
        m.train()
        run_perplexity(m, max_batches=1)
        assert m.training is True
        m.eval()
        run_perplexity(m, max_batches=1)
        assert m.training is False
