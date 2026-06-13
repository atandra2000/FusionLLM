# training/optimizer.py
"""Optimizer: CautiousAdamW + NorMuon (Frozen v1 spec).

Optimizer strategy (per FINAL_FROZEN_SPEC.md §2):
  - NorMuon for 2D weight matrices in MLP/GDN (moe_inter_dim, gdn_d_inner, etc.)
  - CautiousAdamW for embeddings, LM head, MTP projections, norms, biases
  - muon_lr = 0.02, muon_momentum = 0.95
  - adamw_lr = 3.0e-4, betas = (0.9, 0.95)
  - weight_decay = 0.1, cautious_wd = True
  - grad_clip = 1.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ─────────────────────────────────────────────────────────────────────────────
# NorMuon Optimizer
# ─────────────────────────────────────────────────────────────────────────────


class NorMuon(Optimizer):
    """NorMuon — orthogonalized Adam with per-row RMS for matrix params.

    For matrix params (ndim >= 2):
      1. Compute AdamW first/second moments (bias-corrected).
      2. Raw update = -lr * m / (sqrt(v) + eps).
      3. Orthogonalize: normalise each row by its RMS.
      4. Apply weight decay (decoupled, with cautious masking).

    For non-matrix params: falls back to standard AdamW.

    Reference: Keller Jordan, modded-nanogpt speedrun #41-42.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, cautious_wd=cautious_wd,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious = group["cautious_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialise state if needed
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_count = state["step"]

                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_corr1 = 1.0 - beta1 ** step_count
                bias_corr2 = 1.0 - beta2 ** step_count

                # Compute update
                denom = (exp_avg_sq.sqrt() / (bias_corr2 ** 0.5)).add_(eps)
                update = exp_avg / bias_corr1 / denom

                # Orthogonalize for matrix params: normalise each row by RMS
                if p.ndim >= 2:
                    row_norm = update.norm(p=2, dim=-1, keepdim=True)
                    row_rms = row_norm / (update.size(-1) ** 0.5 + eps)
                    update = update / (row_rms + eps)

                # Cautious weight decay
                if wd > 0:
                    if cautious and p.ndim >= 2:
                        mask = (grad * p).sign() == 1.0
                        p.mul_(1.0 - lr * wd * mask.to(p.dtype))
                    else:
                        p.mul_(1.0 - lr * wd)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# CautiousAdamW Optimizer
# ─────────────────────────────────────────────────────────────────────────────


def _cautious_mask(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sign-masked cautious weight decay mask: (grad * weight).sign() == 1.0."""
    g32 = grad.float() if grad.dtype == torch.bfloat16 else grad
    w32 = weight.float() if weight.dtype == torch.bfloat16 else weight
    return (g32 * w32).sign() == 1.0


class CautiousAdamW(Optimizer):
    """AdamW with sign-masked weight decay.

    Standard AdamW for all params. Cautious weight decay only applies
    where gradient and parameter directions agree.

    Frozen v1 spec: betas=(0.9, 0.95), lr=3e-4, weight_decay=0.1.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, cautious_wd=cautious_wd,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious = group["cautious_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialise state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_count = state["step"]

                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_corr1 = 1.0 - beta1 ** step_count
                bias_corr2 = 1.0 - beta2 ** step_count

                # Compute update
                denom = (exp_avg_sq.sqrt() / (bias_corr2 ** 0.5)).add_(eps)
                update = exp_avg / bias_corr1 / denom

                # Decoupled weight decay (with cautious masking)
                if wd > 0:
                    if cautious and p.dim() >= 2:
                        mask = _cautious_mask(grad, p).to(p.dtype)
                        p.mul_(1.0 - lr * wd * mask)
                    else:
                        p.mul_(1.0 - lr * wd)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_optimizers(
    model: nn.Module,
    adamw_lr: float = 3e-4,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    cautious_wd: bool = True,
) -> tuple[NorMuon | None, CautiousAdamW]:
    """Build the optimizer pair: NorMuon for matrix params, CautiousAdamW for rest.

    NorMuon applies to 2D weight matrices in MLP/GDN layers (w1, w2, w3,
    in_proj, out_proj, b_proj, c_proj, dt_proj, g_proj, conv1d).
    CautiousAdamW covers embeddings, LM head, norms, MTP projections,
    gate biases, and all 1D params.

    Args:
        model: The model (FusionLLM or MTP-wrapped).
        adamw_lr: Learning rate for AdamW params.
        muon_lr: Learning rate for NorMuon params.
        muon_momentum: Momentum for NorMuon (uses beta1).
        adamw_betas: Betas for AdamW.
        weight_decay: Weight decay for both optimizers.
        cautious_wd: Enable cautious weight decay.

    Returns:
        (muon_optimizer, adamw_optimizer)
    """
    # Names/patterns excluded from NorMuon
    exclude_patterns = (
        "embed", "head", "norm", "bias", "gate.bias",
        "proj", "A_log", "dt_bias", "D",
    )

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))

        # NorMuon only for 2D weight matrices in MLP/GDN
        is_matrix = p.ndim >= 2
        is_excluded = any(pattern in name.lower() for pattern in exclude_patterns)

        if is_matrix and not is_excluded:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # NorMuon for matrix params
    muon_opt: NorMuon | None = None
    if muon_params:
        muon_opt = NorMuon(
            muon_params,
            lr=muon_lr,
            betas=(muon_momentum, 0.95),
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
        )

    # CautiousAdamW for remaining params
    adamw_opt = CautiousAdamW(
        adamw_params,
        lr=adamw_lr,
        betas=adamw_betas,
        weight_decay=0.0,   # No weight decay for AdamW batch; handled by cautious
        cautious_wd=False,
    )

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"[optim] NorMuon: {len(muon_params)} tensors, {n_muon:,} params, lr={muon_lr}")
    print(f"[optim] CautiousAdamW: {len(adamw_params)} tensors, {n_adamw:,} params, lr={adamw_lr}")

    return muon_opt, adamw_opt
