# models/fusionllm.py
"""FusionLLM-v1: Hybrid MLA + GDN + MoE + MTP (Frozen v1 spec).

Layer schedule (24 layers):
  - 16 MLA layers (indices 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22) → MoE FFN
  -  8 GDN layers (indices 2,5,8,11,14,17,20,23) → Dense FFN

Architecture (per FINAL_FROZEN_SPEC.md):
  - Tied embeddings (vocab=64000, dim=768)
  - 24 layers with alternating MLA/MoE and GDN/DenseFFN
  - MTP (depth=2) auxiliary prediction heads
  - μP initialization
  - Logit softcap (15.0)
  - Pure PyTorch, BF16 compatible, no Triton, no flash attention
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE
from .gdn import GatedDeltaNet


class DenseFFN(nn.Module):
    """Dense SwiGLU FFN (used in GDN layers)."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FusionLLMBlock(nn.Module):
    """One block: either (MLA + MoE) or (GDN + Dense FFN).

    Frozen v1 spec: no gradient checkpointing config toggle at block level;
    uses torch.utils.checkpoint at the module level.
    """

    def __init__(self, config: dict, layer_idx: int, is_gdn: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_gdn = is_gdn

        # Pre-attention norm
        self.norm1 = nn.RMSNorm(config["dim"], eps=1e-6)

        if is_gdn:
            self.attn = GatedDeltaNet(config, layer_idx=layer_idx)
            self.ffn = DenseFFN(config["dim"], config["inter_dim"])
        else:
            self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
            self.ffn = DeepSeekMoE(config)

        # Pre-FFN norm
        self.norm2 = nn.RMSNorm(config["dim"], eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def softcap(logits: torch.Tensor, cap: float = 15.0) -> torch.Tensor:
    """Logit softcap: cap * tanh(x/cap)."""
    return cap * torch.tanh(logits / cap)


def muP_init(model: nn.Module, config: dict) -> None:
    """Apply μP (Maximal Update Parametrisation) initialisation.

    Per FINAL_FROZEN_SPEC.md §6:
      - Residual stream: std = 1 / sqrt(n_layers)
      - Attention / FFN matrices: std = 1 / dim
      - Embeddings: std = 1 / sqrt(dim)
      - Gate-like/scalar params: zero init
    """
    dim = config["dim"]
    n_layers = config["n_layers"]
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    gate_like_keywords = ("gate", "g_proj", "A_log", "dt_bias", "router", "output_head")

    for name, p in model.named_parameters():
        if any(g in name.lower() for g in gate_like_keywords):
            with torch.no_grad():
                p.data.zero_()
            continue
        if p.dim() < 2:
            continue
        with torch.no_grad():
            std = attn_std
            if "embed" in name:
                std = embed_std
            if "head" in name and getattr(model, "tie_embeddings", False):
                std = embed_std
            p.data.normal_(mean=0.0, std=std)


class FusionLLM(nn.Module):
    """FusionLLM-v1: Full model.

    Frozen v1 spec:
      - 24 layers: 16 MLA + 8 GDN (every 3rd, indices [2,5,8,11,14,17,20,23])
      - Tied embeddings (64000 × 768)
      - μP init
      - Logit softcap (15.0)
      - Pure PyTorch, BF16 compatible
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dim = config["dim"]                         # 768
        self.vocab_size = config["vocab_size"]            # 64000
        self.n_layers = config["n_layers"]                # 24
        self.max_seq_len = config["max_seq_len"]          # 4096
        self.tie_embeddings = config.get("tie_embeddings", True)

        # ── Token embedding (tied with LM head) ───────────────────────────
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size, bias=False)
        if self.tie_embeddings:
            self.head.weight = self.embed.weight

        # ── Build layer schedule ─────────────────────────────────────────
        # GDN at indices [2, 5, 8, 11, 14, 17, 20, 23]
        gdn_indices = {2, 5, 8, 11, 14, 17, 20, 23}
        self.layers = nn.ModuleList([
            FusionLLMBlock(config, i, is_gdn=(i in gdn_indices))
            for i in range(self.n_layers)
        ])

        # ── Final norm ───────────────────────────────────────────────────
        self.norm = nn.RMSNorm(self.dim, eps=1e-6)

        # ── Logit softcap ────────────────────────────────────────────────
        self.logit_softcap = config.get("logit_softcap", 15.0)

        # ── Initialisation ───────────────────────────────────────────────
        self._init_weights()
        if config.get("muP", True):
            muP_init(self, config)

    def _init_weights(self) -> None:
        """Standard weight initialisation before μP override."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass (training).

        Args:
            tokens: (B, T) token IDs.
            start_pos: position offset (unused in v1; kept for MTP API compat).

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"

        x = self.embed(tokens)  # (B, T, dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)

        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)

        return logits

    def forward_with_hidden(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning hidden state (for MTP).

        Args:
            tokens: (B, T) token IDs.
            start_pos: position offset (unused in v1; kept for MTP API compat).

        Returns:
            logits: (B, T, vocab_size)
            hidden: (B, T, dim) — pre-head hidden state
        """
        B, T = tokens.shape
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        hidden = self.norm(x)
        logits = self.head(hidden)

        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)

        return logits, hidden

    def get_moe_layers(self) -> list[DeepSeekMoE]:
        """Return all MoE layers for bias update / load balance."""
        moe_layers: list[DeepSeekMoE] = []
        for layer in self.layers:
            if not layer.is_gdn:
                moe_layers.append(layer.ffn)
        return moe_layers


def build_fusionllm(config: dict) -> FusionLLM:
    """Build a FusionLLM-v1 model from a frozen config dict.

    Args:
        config: dict matching FINAL_FROZEN_SPEC.md §1 fields.

    Returns:
        FusionLLM model (not wrapped by MTP).
    """
    return FusionLLM(config)
