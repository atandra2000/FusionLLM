# training/optimizer.py
"""Optimizer: CautiousAdamW + NorMuon (A100 80GB optimized)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer


class NorMuon(Optimizer):
    """NorMuon: orthogonalized Adam with per-row RMS normalization."""

    def __init__(self, params, lr: float = 0.02, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1, cautious_wd: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cautious_wd=cautious_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta1, beta2, eps, wd, cautious = group["lr"], group["betas"][0], group["betas"][1], group["eps"], group["weight_decay"], group["cautious_wd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_count = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step_count
                bias_corr2 = 1.0 - beta2 ** step_count
                denom = (exp_avg_sq.sqrt() / (bias_corr2 ** 0.5)).add_(eps)
                update = exp_avg / bias_corr1 / denom

                if p.ndim >= 2:
                    row_norm = update.norm(p=2, dim=-1, keepdim=True)
                    row_rms = row_norm / (update.size(-1) ** 0.5 + eps)
                    update = update / (row_rms + eps)

                if wd > 0:
                    if cautious and p.ndim >= 2:
                        mask = (grad * p).sign() == 1.0
                        p.mul_(1.0 - lr * wd * mask.to(p.dtype))
                    else:
                        p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)
        return loss


def _cautious_mask(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cautious weight decay mask."""
    g32 = grad.float() if grad.dtype == torch.bfloat16 else grad
    w32 = weight.float() if weight.dtype == torch.bfloat16 else weight
    return (g32 * w32).sign() == 1.0


class CautiousAdamW(Optimizer):
    """AdamW with sign-masked weight decay."""

    def __init__(self, params, lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1, cautious_wd: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cautious_wd=cautious_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta1, beta2, eps, wd, cautious = group["lr"], group["betas"][0], group["betas"][1], group["eps"], group["weight_decay"], group["cautious_wd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_count = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step_count
                bias_corr2 = 1.0 - beta2 ** step_count
                denom = (exp_avg_sq.sqrt() / (bias_corr2 ** 0.5)).add_(eps)
                update = exp_avg / bias_corr1 / denom

                if wd > 0:
                    if cautious and p.dim() >= 2:
                        mask = _cautious_mask(grad, p).to(p.dtype)
                        p.mul_(1.0 - lr * wd * mask)
                    else:
                        p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)
        return loss


def build_optimizers(model: nn.Module, adamw_lr: float = 3e-4, muon_lr: float = 0.02, muon_momentum: float = 0.95, adamw_betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.1, cautious_wd: bool = True) -> tuple[NorMuon | None, CautiousAdamW]:
    """Build NorMuon (2D matrices) + CautiousAdamW (1D / explicit non-matrix params)."""
    # ponytail: exact-name allowlist for params that should NOT go to NorMuon
    # even when they're 2D. MoE experts, MLA/GDN/MoE weight matrices → NorMuon.
    # 1D (norm γ, biases) and explicit non-matrix params → AdamW.
    ADAMW_EXACT_NAMES = {
        "embed.weight",       # tied with head; sparse updates; large embedding
        "head.weight",        # tied with embed
        "norm.weight",        # RMSNorm γ
        "gate.bias",          # MoE gate bias (driven by update_gate_bias, not Adam)
        "A_log",              # GDN log-decay
        "dt_bias",            # GDN dt bias
        "D",                  # GDN per-head skip
    }

    def goes_to_adamw(name: str, p: torch.Tensor) -> bool:
        if p.ndim < 2:
            return True
        # Match against exact full name or as a dot-prefixed suffix
        # (e.g. "embed.weight" matches "embed.weight" and "a.b.embed.weight";
        # "A_log" matches "layers.0.attn.A_log" via the suffix check).
        for entry in ADAMW_EXACT_NAMES:
            if name == entry or name.endswith("." + entry):
                return True
        return False

    muon_params, adamw_params = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (adamw_params if goes_to_adamw(name, p) else muon_params).append(p)

    muon_opt = NorMuon(muon_params, lr=muon_lr, betas=(muon_momentum, 0.95), weight_decay=weight_decay, cautious_wd=cautious_wd) if muon_params else None
    adamw_opt = CautiousAdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=0.0, cautious_wd=False)

    print(f"[optim] NorMuon: {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params, lr={muon_lr}")
    print(f"[optim] CautiousAdamW: {len(adamw_params)} tensors, {sum(p.numel() for p in adamw_params):,} params, lr={adamw_lr}")
    return muon_opt, adamw_opt
