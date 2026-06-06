# training/optimization.py
"""Optimizer implementations and builder.

Contains the Muon optimizer (Newton-Schulz orthogonalized momentum)
and CautiousAdamW (sign-masked weight decay). Also provides the
build_optimizers() factory that creates the optimizer pair.
"""

from __future__ import annotations

import math

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW

from training.normuon import NorMuon
from utils.distributed import is_main_process

if TYPE_CHECKING:
    from training.configs import ConfigBundle


def _zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Newton-Schulz orthogonalization for the Muon optimizer.

    Approximates U @ V.T from the SVD of G (i.e. projects G onto the
    closest semi-orthogonal matrix).  This is the standard 5-iteration
    (a, b, c) = (3.4445, -4.7750, 2.0315) recipe from Keller Jordan's
    NanoGPT-speedrun notes.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer — Newton-Schulz orthogonalized momentum for matrix
    parameters (Keller Jordan 2024).

    Used for ``weight`` tensors with ``ndim >= 2`` that are not the
    embedding, LM head, MTP projections, norms, or the MoE gate.
    Embeddings and the head are still trained with AdamW (this matches
    Keller Jordan's reference recipe).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def _cautious_mask(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute sign mask for cautious weight decay."""
    g32 = grad.float() if grad.dtype == torch.bfloat16 else grad
    w32 = weight.float() if weight.dtype == torch.bfloat16 else weight
    return (g32 * w32).sign() == 1.0


class CautiousAdamW(AdamW):
    """AdamW with sign-masked weight decay.

    The mask is ``(grad * p).sign() == 1.0`` — weight decay only applies
    at positions where the gradient direction agrees with the parameter
    direction. On by default; pass ``cautious_wd=False`` in the param
    group to disable.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            decay = group.get("weight_decay", 0.0)
            cautious = group.get("cautious_wd", True) and decay > 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if cautious and p.dim() >= 2:
                    p.mul_(1.0 - group["lr"] * decay * _cautious_mask(p.grad, p).to(p.dtype))
                else:
                    p.mul_(1.0 - group["lr"] * decay)
        return super().step(closure)


def build_optimizers(
    model: nn.Module, cfg: ConfigBundle
) -> tuple[Muon | NorMuon | None, CautiousAdamW]:
    """Build the optimizer pair.

    When ``cfg.optimizer == "muon_adamw"``: (Muon, CautiousAdamW).
    When ``cfg.optimizer == "normuon_adamw"``: (NorMuon, CautiousAdamW).

    Muon/NorMuon handles matrix parameters (``weight`` with ``ndim >= 2``
    and not the embedding or LM head).  CautiousAdamW handles the rest.
    """
    embed_keyword = "embed"
    head_keyword = "head"

    matrix_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))

        is_matrix = p.ndim >= 2
        is_embed_or_head = (embed_keyword in name.lower()) or (head_keyword in name.lower())

        if is_matrix and not is_embed_or_head:
            matrix_params.append(p)
        else:
            adamw_params.append(p)

    oc = cfg.optim
    use_normuon = oc.optimizer == "normuon_adamw"
    primary_name = "NorMuon" if use_normuon else "Muon"

    if use_normuon:
        primary: Muon | NorMuon | None = (
            NorMuon(
                matrix_params,
                lr=oc.muon_lr,
                betas=oc.adamw_betas,
                weight_decay=oc.weight_decay,
            )
            if matrix_params
            else None
        )
    else:
        primary = (
            Muon(
                matrix_params,
                lr=oc.muon_lr,
                momentum=oc.muon_momentum,
                weight_decay=oc.weight_decay,
            )
            if matrix_params
            else None
        )

    adamw = CautiousAdamW(
        [
            {
                "params": adamw_params,
                "lr": oc.lr,
                "betas": oc.adamw_betas,
                "weight_decay": 0.0,
                "cautious_wd": False,
            },
        ],
        fused=torch.cuda.is_available(),
    )

    if is_main_process():
        n_primary = sum(p.numel() for p in matrix_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(f"[optim] {primary_name}: {len(matrix_params)} tensors, {n_primary:,} params, lr={oc.muon_lr}")
        print(f"[optim] CautiousAdamW: {len(adamw_params)} tensors, {n_adamw:,} params, lr={oc.lr}")

    return primary, adamw


class WarmupCosineDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Simple warmup + cosine decay scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


__all__ = [
    "Muon",
    "CautiousAdamW",
    "build_optimizers",
    "WarmupCosineDecayScheduler",
]
