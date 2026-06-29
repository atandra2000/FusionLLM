# models/fusionllm.py
"""FusionLLM-v1: Hybrid MLA + GDN + MoE + MTP (A100 80GB optimized)."""

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
    """One block: (MLA + MoE) or (GDN + Dense FFN)."""

    def __init__(self, config: dict, layer_idx: int, is_gdn: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_gdn = is_gdn
        self.use_checkpoint = not is_gdn

        self.norm1 = nn.RMSNorm(config["dim"], eps=1e-6)

        if is_gdn:
            self.attn = GatedDeltaNet(config, layer_idx=layer_idx)
            self.ffn = DenseFFN(config["dim"], config["inter_dim"])
        else:
            self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
            self.ffn = DeepSeekMoE(config)

        self.norm2 = nn.RMSNorm(config["dim"], eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def softcap(logits: torch.Tensor, cap: float = 15.0) -> torch.Tensor:
    """Logit softcap."""
    return cap * torch.tanh(logits / cap)


def muP_init(model: nn.Module, config: dict) -> None:
    """Apply μP initialization."""
    dim = config["dim"]
    n_layers = config["n_layers"]
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    gate_keywords = ("gate", "g_proj", "A_log", "dt_bias", "router", "output_head")

    for name, p in model.named_parameters():
        if any(g in name.lower() for g in gate_keywords):
            with torch.no_grad():
                p.data.zero_()
            continue
        if p.dim() < 2:
            continue
        with torch.no_grad():
            std = embed_std if "embed" in name or ("head" in name and getattr(model, "tie_embeddings", False)) else attn_std
            p.data.normal_(mean=0.0, std=std)


class FusionLLM(nn.Module):
    """FusionLLM-v1: Full model (24 layers: 16 MLA + 8 GDN)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dim = config["dim"]
        self.vocab_size = config["vocab_size"]
        self.n_layers = config["n_layers"]
        self.max_seq_len = config["max_seq_len"]
        self.tie_embeddings = config.get("tie_embeddings", True)

        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size, bias=False)
        if self.tie_embeddings:
            self.head.weight = self.embed.weight

        gdn_indices = {2, 5, 8, 11, 14, 17, 20, 23}
        self.layers = nn.ModuleList([FusionLLMBlock(config, i, is_gdn=(i in gdn_indices)) for i in range(self.n_layers)])
        self.norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.logit_softcap = config.get("logit_softcap", 15.0)

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
        """Forward pass (training)."""
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"

        x = self.embed(tokens)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False) if layer.use_checkpoint else layer(x)
        x = self.norm(x)
        logits = self.head(x)

        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)
        return logits

    def forward_with_hidden(self, tokens: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning hidden state (for MTP)."""
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        hidden = self.norm(x)
        logits = self.head(hidden)
        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)
        return logits, hidden

    def get_moe_layers(self) -> list[DeepSeekMoE]:
        """Return all MoE layers."""
        return [layer.ffn for layer in self.layers if not layer.is_gdn]


def build_fusionllm(config: dict) -> FusionLLM:
    """Build FusionLLM model."""
    return FusionLLM(config)
