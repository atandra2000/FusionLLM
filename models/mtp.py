# models/mtp.py
"""Multi-Token Prediction (auxiliary heads for future-token supervision)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla import MultiHeadLatentAttention
from .moe import SwiGLUExpert


def softcap_ce(logits: torch.Tensor, target: torch.Tensor, cap: float = 15.0, ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy with logit softcap."""
    logits = cap * torch.tanh(logits / cap)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=ignore_index, reduction="mean")


class MTPTransformerBlock(nn.Module):
    """MTP transformer block: pre-norm, MLA, SwiGLU FFN."""

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
    """Single MTP prediction head."""

    def __init__(self, config: dict, depth: int):
        super().__init__()
        self.depth = depth
        dim = config["dim"]
        self.norm_h = nn.RMSNorm(dim, eps=1e-6)
        self.norm_e = nn.RMSNorm(dim, eps=1e-6)
        # ponytail: was proj/proj_aux split gated on depth, but both branches
        # built the exact same nn.Linear(2*dim, dim, bias=False) and forward
        # used whichever was non-None — collapse to one projection.
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.block = MTPTransformerBlock(config)
        self.norm_out = nn.RMSNorm(dim, eps=1e-6)
        self.output_head: nn.Linear | None = None

    def set_output_head(self, head: nn.Linear) -> None:
        self.output_head = head

    def forward(self, hidden: torch.Tensor, target_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if self.output_head is None:
            raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")
        fused = self.proj(torch.cat([self.norm_h(hidden), self.norm_e(target_emb)], dim=-1))
        h = self.block(fused)
        return self.output_head(self.norm_out(h)), h


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction wrapper."""

    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.config = config
        self.main_model = main_model
        self.depth = config.get("mtp_depth", 2)
        self.softcap = config.get("mtp_softcap", True)
        self.softcap_value = config.get("mtp_softcap_value", 15.0)
        self.mtp_loss_weights = config.get("mtp_loss_weights", [config.get("mtp_loss_weight_1", 0.10), config.get("mtp_loss_weight_2", 0.05)])
        # Validate up front so the IndexError doesn't surface on the first forward.
        if self.depth > 0 and len(self.mtp_loss_weights) < self.depth:
            raise ValueError(
                f"mtp_loss_weights has {len(self.mtp_loss_weights)} entries but "
                f"mtp_depth={self.depth} requires at least {self.depth}. "
                f"Pass `mtp_loss_weights=[w1, w2, ...]` (one per depth) or set "
                f"mtp_depth <= len(mtp_loss_weights)."
            )
        self.embed = main_model.embed
        if self.depth > 0:
            self.mtp_modules = nn.ModuleList([MTPModule(config, d + 1) for d in range(self.depth)])
            shared_head = main_model.head
            for mtp in self.mtp_modules:
                mtp.set_output_head(shared_head)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Run main model + MTP heads."""
        B, T = tokens.shape
        main_logits, main_hidden = self.main_model.forward_with_hidden(tokens, start_pos)
        mtp_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        prev_hidden = main_hidden
        for d in range(self.depth):
            mtp, depth = self.mtp_modules[d], d + 1
            usable = T - depth - 1
            if usable <= 0:
                break
            h_in = prev_hidden[:, :usable]
            target_emb = self.embed(tokens[:, depth:depth + usable])
            targets = tokens[:, depth + 1:depth + 1 + usable]
            logits, new_hidden = mtp(h_in, target_emb)
            mtp_outputs.append((logits, targets, torch.tensor(self.mtp_loss_weights[d], device=logits.device)))
            prev_hidden = new_hidden
        return main_logits, mtp_outputs

    def compute_mtp_loss(self, mtp_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ignore_index: int = -100) -> torch.Tensor:
        """Compute weighted MTP loss."""
        if not mtp_outputs:
            return torch.tensor(0.0)
        total_loss = torch.tensor(0.0, device=mtp_outputs[0][0].device)
        for logits, targets, weight in mtp_outputs:
            loss = softcap_ce(logits, targets, cap=self.softcap_value, ignore_index=ignore_index) if self.softcap else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index, reduction="mean")
            total_loss = total_loss + weight * loss
        return total_loss
