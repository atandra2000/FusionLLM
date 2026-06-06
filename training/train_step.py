# training/train_step.py
"""Training step execution.

Handles the forward pass, loss computation, backward pass, and
optimizer step for a single training micro-step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from training.numerical_health import NumericalHealthMonitor
from utils.distributed import all_reduce_mean
from utils.tensor_checks import validate_loss

if TYPE_CHECKING:
    from training.configs import ConfigBundle
    from training.optimization import Muon, CautiousAdamW


def compute_loss(
    model: torch.nn.Module,
    mtp: torch.nn.Module | None,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    cfg: ConfigBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the total loss (CE + balance + z-loss).

    Args:
        model: The main model (or MTP wrapper)
        mtp: Optional MTP wrapper
        tokens: Input tokens
        targets: Target tokens
        cfg: ConfigBundle with loss weights
        device: Device for tensors

    Returns:
        (total_loss, ce_loss, balance_loss, z_loss)
    """
    if mtp is not None:
        main_logits, mtp_pairs, _ = mtp(tokens)
        ce_loss = F.cross_entropy(
            main_logits.reshape(-1, main_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        if mtp_pairs:
            mtp_loss = mtp.compute_mtp_loss(mtp_pairs)
        else:
            mtp_loss = ce_loss.new_zeros(())
        main_loss = ce_loss
        ce_loss = main_loss + mtp_loss
    else:
        logits = model(tokens, start_pos=0, use_cache=False)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

    # MoE balance loss
    balance_loss = torch.zeros((), device=device)
    if hasattr(model, 'moe_layers'):
        losses = [moe.get_load_balance_loss() for moe in model.moe_layers()]
        if losses:
            balance_loss = torch.stack(losses).sum()

    # Router z-loss
    z_loss = torch.zeros((), device=device)
    if hasattr(model, 'moe_layers'):
        moe_layers = list(model.moe_layers())
        for moe in moe_layers:
            z_loss = z_loss + moe.get_z_loss()
        z_loss = z_loss / max(1, len(moe_layers))

    loss = (
        ce_loss
        + cfg.balance_loss_alpha * balance_loss
        + cfg.z_loss_weight * z_loss
    ) / cfg.data.gradient_accumulation_steps

    return loss, ce_loss, balance_loss, z_loss


def optimizer_step(
    raw_model: torch.nn.Module,
    muon: Muon | None,
    adamw: CautiousAdamW,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: ConfigBundle,
) -> float:
    """Execute a single optimizer step with gradient clipping.

    Args:
        raw_model: The unwrapped model
        muon: Optional Muon optimizer
        adamw: CautiousAdamW optimizer
        scaler: GradScaler for mixed precision
        scheduler: Learning rate scheduler
        cfg: ConfigBundle with optimizer settings

    Returns:
        Gradient norm value
    """
    params = [p for p in raw_model.parameters() if p.requires_grad]

    # Unscale gradients for clipping
    scaler.unscale_(adamw)
    if muon is not None:
        scaler.unscale_(muon)

    # Compute gradient norm BEFORE clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.max_grad_norm)

    if muon is not None:
        scaler.step(muon)
    scaler.step(adamw)
    scaler.update()

    scheduler.step()

    if muon is not None:
        muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)

    # Refresh MoE weight stacks after optimizer step
    if hasattr(raw_model, 'moe_layers'):
        for moe in raw_model.moe_layers():
            if hasattr(moe, '_refresh_weight_stacks'):
                moe._refresh_weight_stacks()

    return grad_norm.item()


def train_step(
    model: torch.nn.Module,
    mtp: torch.nn.Module | None,
    raw_model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    micro_step: int,
    cfg: ConfigBundle,
    muon: Muon | None,
    adamw: CautiousAdamW,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    health_monitor: NumericalHealthMonitor,
    device: torch.device,
    rank: int,
    amp_context,
) -> dict[str, float]:
    """Execute a single training step.

    Args:
        model: The main model (or MTP wrapper)
        mtp: Optional MTP wrapper
        raw_model: Unwrapped model for MoE updates
        tokens: Input tokens
        targets: Target tokens
        micro_step: Current micro step
        cfg: ConfigBundle
        muon: Optional Muon optimizer
        adamw: CautiousAdamW optimizer
        scaler: GradScaler
        scheduler: LR scheduler
        health_monitor: NumericalHealthMonitor
        device: Device
        rank: Process rank
        amp_context: AMP context manager

    Returns:
        Dictionary of metrics
    """
    is_opt_step = (micro_step + 1) % cfg.data.gradient_accumulation_steps == 0

    with amp_context:
        loss, ce_loss, balance_loss, z_loss = compute_loss(
            model, mtp, tokens, targets, cfg, device
        )

    # NaN/Inf check on loss before backward
    validate_loss(loss, step=micro_step)

    # Numerical health monitoring
    if health_monitor.is_active:
        ce_scalar = all_reduce_mean(ce_loss.detach())
        if rank == 0:
            health_monitor.update_loss(ce_scalar.item(), micro_step)

    scaler.scale(loss).backward()

    if is_opt_step:
        if health_monitor.is_active:
            health_monitor.update_gradients(raw_model, micro_step)

        grad_norm = optimizer_step(
            raw_model, muon, adamw, scaler, scheduler, cfg
        )

    ce_scalar = all_reduce_mean(ce_loss.detach())
    balance_scalar = all_reduce_mean(balance_loss.detach())
    z_loss_scalar = all_reduce_mean(z_loss.detach())
    return {
        "loss": ce_scalar.item(),
        "balance_loss": balance_scalar.item(),
        "z_loss": z_loss_scalar.item(),
    }


__all__ = [
    "compute_loss",
    "optimizer_step",
    "train_step",
]
