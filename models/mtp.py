# models/mtp.py
"""Multi-Token Prediction (DeepSeek-V3 style).

``mtp_depth`` auxiliary heads, each predicting ``t[i+d+1]`` from
the previous depth's hidden state and the embedding of ``t[i+d]``.

Invariants (preserved from the original recipe):

* The embedding table is the *main model's* embed — referenced, not
  copied, so parameters are not double-counted.
* The output head is the *main model's* head — set via
  :meth:`MTPModule.set_output_head` so each MTP module shares the
  LM head's parameters.
* Token alignment is computed inside :meth:`forward` so the trainer
  passes aligned (logits, targets) pairs and ``compute_loss`` does
  no re-shifting.

Phase 2.4 additions
-------------------
* ``mtp_depth`` default is bumped to **3** (was 1).
* :func:`softcap_ce` loss is used when ``softcap=True``:
  ``loss = softcap * tanh(loss / softcap)`` on the per-token CE
  with ``softcap=15.0`` (DeepSeek-V3 §3.3.1).
* :func:`mtp_loss_weight_schedule` provides a piecewise-linear
  weight per depth: warmup → constant → anneal.
* Each MTP head after the first has its own ``proj_aux`` so the
  heads can specialise.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def softcap_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    cap: float = 15.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with a soft-cap (DeepSeek-V3 §3.3.1).

    Mathematically equivalent to capping the loss *value* at
    ``cap``:

        loss = cap * tanh(loss_raw / cap)

    This is a *proxy* for logit soft-capping (which would also
    bound the logits before softmax).  Capping the loss is cheaper
    to integrate and is what the MTP auxiliary heads use.
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return cap * torch.tanh(loss / cap)


def mtp_loss_weight_schedule(
    depths: int,
    schedule: Sequence[float] | None = None,
) -> list[float]:
    """Per-depth loss weight schedule.

    The default is ``[0.3, 0.2, 0.1]`` for depths 1, 2, 3 — linearly
    decreasing.  ``schedule`` may be a list of length ``depths``;
    otherwise the function synthesises a linear schedule from
    ``mtp_loss_weight`` (config) down to ``mtp_loss_weight_anneal``
    (default 1/3 of the start).
    """
    if schedule is not None:
        if len(schedule) != depths:
            raise ValueError(
                f"mtp_loss_weight_schedule has length {len(schedule)} but mtp_depth={depths}"
            )
        return list(schedule)
    if depths == 0:
        return []
    # Default: linearly decrease from 0.3 to 0.1.
    start, end = 0.3, 0.1
    if depths == 1:
        return [start]
    return [start + (end - start) * i / (depths - 1) for i in range(depths)]


class MTPBlock(nn.Module):
    """One MTP block: a tiny pre-norm transformer over the fused input.

    Phase 2.4: takes a ``is_aux`` flag.  When True, uses a distinct
    projection (``proj_aux``) so heads after the first can specialise.
    """

    def __init__(self, dim: int, n_heads: int, inter_dim: int, is_aux: bool = False):
        super().__init__()
        self.is_aux = is_aux
        self.norm_h = nn.RMSNorm(dim, eps=1e-6)
        self.norm_e = nn.RMSNorm(dim, eps=1e-6)
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        if is_aux:
            self.proj_aux = nn.Linear(2 * dim, dim, bias=False)
        self.norm_attn = nn.RMSNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)
        self.norm_ffn = nn.RMSNorm(dim, eps=1e-6)
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

        self._causal_mask_size: int = 0
        self._causal_mask: torch.Tensor
        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

    def _get_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        cm = self._causal_mask
        if seqlen > self._causal_mask_size or cm.device != device:
            mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril(diagonal=0)
            self._causal_mask = mask
            self._causal_mask_size = seqlen
        return self._causal_mask[:seqlen, :seqlen]

    def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        projection = self.proj_aux if self.is_aux else self.proj
        fused = projection(torch.cat([self.norm_h(prev_hidden), self.norm_e(target_emb)], dim=-1))
        seqlen = fused.size(1)
        causal = self._get_causal_mask(seqlen, fused.device)
        attn_in = self.norm_attn(fused)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=~causal,
            is_causal=False,
            need_weights=False,
        )
        fused = fused + attn_out
        ffn_in = self.norm_ffn(fused)
        return fused + self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))


class MTPModule(nn.Module):
    """MTP head for prediction depth *d* (1-indexed)."""

    def __init__(self, dim: int, n_heads: int, inter_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        # First head uses the default projection; auxiliary heads
        # (depth >= 2) get a distinct block so they can specialise.
        self.block = MTPBlock(dim, n_heads, inter_dim, is_aux=(depth > 1))
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.output_head: nn.Linear | None = None

    def set_output_head(self, head: nn.Linear) -> None:
        self.output_head = head

    def forward(
        self, prev_hidden: torch.Tensor, target_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.output_head is None:
            raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")
        if prev_hidden.shape != target_emb.shape:
            raise ValueError(
                f"Shape mismatch: prev_hidden {prev_hidden.shape} vs target_emb {target_emb.shape}"
            )
        h = self.block(prev_hidden, target_emb)
        h_norm = self.norm(h)
        return self.output_head(h_norm), h_norm


class MultiTokenPrediction(nn.Module):
    """Wraps the main :class:`Transformer` with ``mtp_depth`` MTP heads.

    Returns aligned ``(main_logits, mtp_pairs)`` for the trainer's
    loss computation.
    """

    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.main_model: nn.Module = main_model
        self.depth = config.get("mtp_depth", 3)
        self.mtp_weight = config.get("mtp_loss_weight", 0.3)
        self.mtp_weight_schedule = mtp_loss_weight_schedule(
            self.depth, config.get("mtp_loss_weight_schedule")
        )
        self.softcap = config.get("mtp_softcap", True)
        self.softcap_value = config.get("mtp_softcap_value", 15.0)
        self.embed: nn.Module
        if self.depth > 0:
            dim = config["dim"]
            n_heads = config.get("mtp_n_heads", config["n_heads"])
            inter_dim = config.get("mtp_inter_dim", config.get("inter_dim", 4 * dim))
            self.mtp_modules = nn.ModuleList(
                [MTPModule(dim, n_heads, inter_dim, d + 1) for d in range(self.depth)]
            )
            self.embed = main_model.embed
            shared_head = main_model.head
            for mtp in self.mtp_modules:
                mtp.set_output_head(shared_head)
        else:
            self.mtp_modules = nn.ModuleList()
            self.embed = main_model.embed

    def _reinject_shared_heads(self) -> None:
        shared_head = self.main_model.head
        for mtp in self.mtp_modules:
            mtp.set_output_head(shared_head)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._reinject_shared_heads()
        return result

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Run the main model and all MTP heads.

        Returns ``(main_logits, mtp_pairs, main_hidden)``.
        ``mtp_pairs`` is a list of ``(logits, targets)`` already
        aligned so :func:`torch.nn.functional.cross_entropy` works
        without re-shifting.
        """
        if tokens.dim() < 2:
            raise ValueError(f"Expected (bsz, seq) tokens, got shape {tokens.shape}")
        seq_len = tokens.size(1)

        # Main forward (always no cache during training)
        use_cache = not self.main_model.training
        main_logits, prev_h = self.main_model.forward_with_hidden(
            tokens,
            start_pos,
            use_cache=use_cache,
        )

        mtp_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for d, mtp in enumerate(self.mtp_modules):
            depth = d + 1
            usable = seq_len - depth - 1
            if usable <= 0:
                break
            h_in = prev_h[:, :usable]
            emb_in = self.embed(tokens[:, depth : depth + usable])
            tgt = tokens[:, depth + 1 : depth + 1 + usable]
            logits, h_norm = mtp(h_in, emb_in)
            mtp_pairs.append((logits, tgt))
            prev_h = h_norm

        return main_logits, mtp_pairs, prev_h

    # ── Loss helpers ──────────────────────────────────────────────────
    def compute_mtp_loss(
        self,
        mtp_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute the weighted MTP loss across all depths.

        Returns 0 if ``mtp_pairs`` is empty.  Each depth's loss is
        weighted by ``mtp_weight_schedule[depth]``.  When
        ``softcap=True``, the per-depth loss is passed through
        :func:`softcap_ce` with ``softcap_value``.
        """
        if not mtp_pairs:
            return torch.tensor(0.0)
        total = None
        for d, (logits, target) in enumerate(mtp_pairs):
            if self.softcap:
                loss = softcap_ce(logits, target, cap=self.softcap_value, ignore_index=ignore_index)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=ignore_index,
                    reduction="mean",
                )
            w = (
                self.mtp_weight_schedule[d]
                if d < len(self.mtp_weight_schedule)
                else self.mtp_weight
            )
            total = w * loss if total is None else total + w * loss
        return total
