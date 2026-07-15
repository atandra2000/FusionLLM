# training/validation.py
"""Validation loss and perplexity on synthetic data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic synthetic batch.

    Ponytail: same uniform-random data as before, but seeded so a second
    call (e.g. next eval) produces a different batch deterministically
    rather than re-sampling from the live RNG. The batch is still
    uniform-random, so an untrained model will hit ~ln(vocab_size); the
    point of seeding is reproducibility, not realism. Replace with a real
    held-out corpus once the data pipeline exists.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return tokens.to(device), targets.to(device)


@torch.no_grad()
def compute_validation_loss(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 4096,
    vocab_size: int = 64000,
    num_batches: int = 8,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Compute validation loss and perplexity on synthetic data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        tokens, targets = generate_synthetic_batch(
            batch_size, seq_len, vocab_size, device
        )

        logits = model(tokens)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += batch_size * seq_len

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    model.train()

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "n_tokens": total_tokens,
    }
