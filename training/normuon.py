# training/normuon.py
"""NorMuon optimizer — orthogonalized Adam with per-row RMS.

NorMuon (modded-nanogpt #41–42) combines AdamW's per-parameter adaptive
learning rates with orthogonalization of the update direction for matrix
parameters.  Instead of the Newton-Schulz iteration used by Muon, it
normalises each row of the update by its RMS, which is cheaper and
numerically simpler.

Algorithm for matrix params (ndim >= 2):
  1. Compute AdamW first/second moments (bias-corrected).
  2. Compute raw update = -lr * m / (sqrt(v) + eps).
  3. Orthogonalize: normalise each row of the update by its RMS.
  4. Apply weight decay (decoupled, optional cautious masking).

For non-matrix params: falls back to standard AdamW (no orthogonalization).

Reference
---------
Keller Jordan, "modded-nanogpt speedrun", records #41–42 (April 2026).
"""

from __future__ import annotations

import math
import warnings
import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional


def validate_normuon_config(
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> List[str]:
    """Validate NorMuon configuration and return warnings.
    
    Args:
        lr: Learning rate
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: Weight decay
        
    Returns:
        List of warning messages (empty if valid)
    """
    warnings_list = []
    
    # Learning rate validation
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    if lr > 1.0:
        warnings_list.append(f"Learning rate {lr} is unusually high; typical range is 1e-5 to 1e-3")
    
    # Betas validation
    if not (0.0 <= betas[0] < 1.0):
        raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
    if not (0.0 <= betas[1] < 1.0):
        raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")
    if betas[0] >= betas[1]:
        warnings_list.append(f"beta1 ({betas[0]}) >= beta2 ({betas[1]}); typically beta1 < beta2")
    
    # Epsilon validation
    if eps <= 0:
        raise ValueError(f"Epsilon must be positive, got {eps}")
    if eps > 1e-3:
        warnings_list.append(f"Epsilon {eps} is unusually large; typical range is 1e-8 to 1e-6")
    
    # Weight decay validation
    if weight_decay < 0:
        raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
    if weight_decay > 1.0:
        warnings_list.append(f"Weight decay {weight_decay} is unusually high; typical range is 0.01 to 0.3")
    
    return warnings_list


def validate_param_groups(
    param_groups: List[Dict],
    model: Optional[torch.nn.Module] = None,
) -> List[str]:
    """Validate parameter groups for NorMuon.
    
    Args:
        param_groups: List of parameter group dicts
        model: Optional model for additional checks
        
    Returns:
        List of warning messages
    """
    warnings_list = []
    
    if not param_groups:
        raise ValueError("No parameter groups provided")
    
    total_params = 0
    matrix_params = 0
    scalar_params = 0
    
    for i, group in enumerate(param_groups):
        params = group.get("params", [])
        if not params:
            warnings_list.append(f"Parameter group {i} has no parameters")
            continue
        
        for p in params:
            total_params += 1
            if p.ndim >= 2:
                matrix_params += 1
            else:
                scalar_params += 1
    
    if total_params == 0:
        raise ValueError("No parameters found in parameter groups")
    
    # Check for NorMuon-specific concerns
    if matrix_params == 0:
        warnings_list.append("No matrix parameters found; NorMuon orthogonalization will not be applied")
    
    if scalar_params > matrix_params:
        warnings_list.append(
            f"More scalar params ({scalar_params}) than matrix params ({matrix_params}); "
            "consider using AdamW for scalar params"
        )
    
    return warnings_list


class NorMuon(Optimizer):
    """NorMuon — orthogonalized Adam with per-row RMS for matrix params.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate.
        betas: AdamW betas (first/second moment decay).
        eps: AdamW epsilon for numerical stability.
        weight_decay: decoupled weight decay.
        cautious_wd: apply sign-masked cautious weight decay for matrix
            params (modded-nanogpt #43).
        validate: enable configuration validation (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
        validate: bool = True,
    ):
        # Validate configuration
        if validate:
            config_warnings = validate_normuon_config(lr, betas, eps, weight_decay)
            for w in config_warnings:
                warnings.warn(w, UserWarning, stacklevel=2)
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
        )
        super().__init__(params, defaults)
        
        # Validate parameter groups
        if validate:
            param_warnings = validate_param_groups(self.param_groups)
            for w in param_warnings:
                warnings.warn(w, UserWarning, stacklevel=2)

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

                denom = (exp_avg_sq.sqrt() / bias_corr2 ** 0.5).add_(eps)
                update = exp_avg / bias_corr1 / denom

                if p.ndim >= 2:
                    row_rms = update.norm(p=2, dim=-1, keepdim=True)
                    row_rms = row_rms / (update.size(-1) ** 0.5 + eps)
                    update = update / (row_rms + eps)

                if wd > 0:
                    if cautious and p.ndim >= 2:
                        mask = (grad * p).sign() == 1.0
                        p.mul_(1.0 - lr * wd * mask.to(p.dtype))
                    else:
                        p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)
        return loss
    
    def get_config_summary(self) -> Dict:
        """Get summary of optimizer configuration."""
        if not self.param_groups:
            return {}
        
        group = self.param_groups[0]
        return {
            "lr": group["lr"],
            "betas": group["betas"],
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
            "cautious_wd": group["cautious_wd"],
            "num_param_groups": len(self.param_groups),
            "total_params": sum(len(g["params"]) for g in self.param_groups),
        }
