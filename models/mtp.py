# models/mtp.py
"""Multi-Token Prediction (Frozen v1 spec).

Architecture (per FINAL_FROZEN_SPEC.md §5.6):
  MTP Module 1:
    ├─ proj: Linear(2×768→768)
    │     input = concat(main_hidden[t], embed[t+1])
    ├─ SharedTransformerBlock (pre-norm, MLA, dense FFN)
    ├─ norm + tied output head → logits_1
    └─ Target: tokens[t+2]
    Weight = 0.10

  MTP Module 2:
    ├─ proj_aux: Linear(2×768→768)
    ├─ SharedTransformerBlock (pre-norm, MLA, dense FFN)
    ├─ norm + tied output head → logits_2
    └─ Target: tokens[t+3]
    Weight = 0.05

Pure PyTorch, BF16 compatible. Tied head to main embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla import MultiHeadLatentAttention
from .moe import SwiGLUExpert


def softcap_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    cap: float = 15.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with soft-cap on logits: cap * tanh(x/cap)."""
    logits = cap * torch.tanh(logits / cap)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


class MTPTransformerBlock(nn.Module):
    """Shared transformer block for MTP: pre-norm, MLA, dense FFN (SwiGLU)."""

    def __init__(self, config: dict):
        super().__init__()
        self.norm1 = nn.RMSNorm(config["dim"], eps=1e-6)
        self.attn = MultiHeadLatentAttention(config)
        self.norm2 = nn.RMSNorm(config["dim"], eps=1e-6)
        self.ffn = SwiGLUExpert(config["dim"], config["mtp_inter_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MTPModule(nn.Module):
    """Single MTP prediction head.

    Each depth has its own projection + transformer block + norm.
    Output head is tied to the main model's embedding.
    """

    def __init__(self, config: dict, depth: int):
        super().__init__()
        self.depth = depth
        dim = config["dim"]
        self.norm_h = nn.RMSNorm(dim, eps=1e-6)
        self.norm_e = nn.RMSNorm(dim, eps=1e-6)
        # Depth 1 uses proj; depth >= 2 uses proj_aux (budget counts only one per depth)
        if depth >= 2:
            self.proj = None
            self.proj_aux = nn.Linear(2 * dim, dim, bias=False)
        else:
            self.proj = nn.Linear(2 * dim, dim, bias=False)
            self.proj_aux = None
        self.block = MTPTransformerBlock(config)
        self.norm_out = nn.RMSNorm(dim, eps=1e-6)
        # Output head (tied, set externally)
        self.output_head: nn.Linear | None = None

    def set_output_head(self, head: nn.Linear) -> None:
        self.output_head = head

    def forward(
        self,
        hidden: torch.Tensor,  # (B, T, dim)
        target_emb: torch.Tensor,  # (B, T, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: (B, T, vocab_size)
            new_hidden: (B, T, dim) — output of the transformer block, before norm
        """
        if self.output_head is None:
            raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")
        # Fuse hidden state and target embedding
        proj = self.proj_aux if self.proj_aux is not None else self.proj
        fused = proj(torch.cat([self.norm_h(hidden), self.norm_e(target_emb)], dim=-1))
        h = self.block(fused)
        h_norm = self.norm_out(h)
        return self.output_head(h_norm), h


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction wrapper.

    Wraps the main model and adds mtp_depth auxiliary prediction heads.
    Each head predicts t[d+1] steps ahead using the previous head's hidden
    state and the target token's embedding.

    Frozen v1 spec:
      - mtp_depth = 2
      - mtp_loss_weight_1 = 0.10, mtp_loss_weight_2 = 0.05
      - mtp_inter_dim = 2048
      - mtp_softcap = True, mtp_softcap_value = 15.0
      - mtp_share_attention = True
      - mtp_tied_head = True
    """

    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.config = config
        self.main_model = main_model
        self.depth = config.get("mtp_depth", 2)
        self.softcap = config.get("mtp_softcap", True)
        self.softcap_value = config.get("mtp_softcap_value", 15.0)

        # Loss weights (frozen)
        raw_weights = config.get("mtp_loss_weights", None)
        if raw_weights is not None:
            self.mtp_loss_weights = raw_weights
        else:
            self.mtp_loss_weights = [
                config.get("mtp_loss_weight_1", 0.10),
                config.get("mtp_loss_weight_2", 0.05),
            ]

        # Share the main model's embedding table
        self.embed = main_model.embed

        # Create MTP modules
        if self.depth > 0:
            self.mtp_modules = nn.ModuleList([
                MTPModule(config, d + 1) for d in range(self.depth)
            ])
            # Tie output heads to main model's head
            shared_head = main_model.head
            for mtp in self.mtp_modules:
                mtp.set_output_head(shared_head)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Run main model forward + all MTP heads.

        Args:
            tokens: (B, T) token IDs.

        Returns:
            main_logits: (B, T, vocab_size)
            mtp_outputs: list of (logits, targets, weights) for each depth.
        """
        B, T = tokens.shape

        # Main model forward
        main_logits, main_hidden = self.main_model.forward_with_hidden(tokens, start_pos)

        mtp_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        prev_hidden = main_hidden  # (B, T, dim)

        for d in range(self.depth):
            mtp = self.mtp_modules[d]
            depth = d + 1

            # Shift: need input = prev_hidden[t], target_emb = embed(tokens[t+depth])
            # Target = tokens[t+depth+1]
            # We use positions [0, T-depth-1) for the MTP inputs
            usable = T - depth - 1
            if usable <= 0:
                break

            # Inputs
            h_in = prev_hidden[:, :usable]                     # (B, usable, dim)
            target_emb = self.embed(tokens[:, depth : depth + usable])  # (B, usable, dim)
            targets = tokens[:, depth + 1 : depth + 1 + usable]  # (B, usable)

            # Forward through MTP module
            logits, new_hidden = mtp(h_in, target_emb)  # (B, usable, vocab), (B, usable, dim)

            mtp_outputs.append((logits, targets, torch.tensor(self.mtp_loss_weights[d], device=logits.device)))
            prev_hidden = new_hidden

        return main_logits, mtp_outputs

    def compute_mtp_loss(
        self,
        mtp_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute weighted MTP loss across all depths."""
        if not mtp_outputs:
            return torch.tensor(0.0)
        total_loss = torch.tensor(0.0, device=mtp_outputs[0][0].device)
        for logits, targets, weight in mtp_outputs:
            if self.softcap:
                loss = softcap_ce(logits, targets, cap=self.softcap_value, ignore_index=ignore_index)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=ignore_index,
                    reduction="mean",
                )
            total_loss = total_loss + weight * loss
        return total_loss
