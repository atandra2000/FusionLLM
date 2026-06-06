# training/validation.py
"""Evaluation logic for the training pipeline.

Handles perplexity evaluation and task-based evaluation during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.distributed import is_main_process

if TYPE_CHECKING:
    import torch
    from training.configs import EvalConfig


def maybe_eval(
    step: int,
    eval_cfg: EvalConfig,
    raw_model: torch.nn.Module,
    device: torch.device,
    logger,
    runs_csv,
    log_fn,
) -> None:
    """Run evaluation if enabled and at the right step.

    Args:
        step: Current training step
        eval_cfg: Evaluation configuration
        raw_model: The unwrapped model
        device: Device for evaluation
        logger: TrainerLogger for logging metrics
        runs_csv: RunsCsvLogger for CSV logging
        log_fn: Logging function for messages
    """
    if not eval_cfg.eval_enabled:
        return
    if step <= 0 or step % eval_cfg.eval_interval != 0:
        return
    if not is_main_process():
        return

    from eval.eval_core import make_synthetic_loader, run_perplexity

    loader = make_synthetic_loader() if eval_cfg.eval_synthetic else None
    ppl_metrics: dict[str, float] = {}
    try:
        ppl_metrics = run_perplexity(
            raw_model,
            loader,
            device=str(device),
            max_batches=eval_cfg.eval_max_batches,
        )
    except Exception as exc:
        log_fn(f"[eval] step {step}: perplexity failed: {exc!r}")

    if ppl_metrics:
        log_fn(
            f"[eval] step={step} val_loss={ppl_metrics['loss']:.4f} "
            f"val_ppl={ppl_metrics['ppl']:.2f} n_tokens={int(ppl_metrics['n_tokens'])}"
        )

    eval_task_metrics: dict[str, float] = {}
    if not eval_cfg.eval_synthetic:
        from eval.run_lm_eval import run_lm_eval

        try:
            task_results = run_lm_eval(
                raw_model,
                tasks=eval_cfg.eval_tasks,
                device=str(device),
                limit=50,
            )
            if task_results is not None:
                for task_name, score in task_results.items():
                    log_fn(f"[eval]   {task_name}: {score:.4f}")
                    eval_task_metrics[task_name] = score
        except Exception as exc:
            log_fn(f"[eval] step {step}: lm_eval failed: {exc!r}")

    val_metrics = dict(ppl_metrics)
    val_metrics.pop("n_tokens", None)
    val_metrics.update(eval_task_metrics)
    if val_metrics:
        try:
            logger.log_validation(
                step,
                ppl_metrics.get("loss", 0.0),
                val_metrics=val_metrics,
            )
        except Exception as exc:
            log_fn(f"[eval] step {step}: logger rejected metrics: {exc!r}")

    try:
        runs_csv.log(
            step,
            loss=ppl_metrics.get("loss", 0.0),
            ppl=ppl_metrics.get("ppl", 0.0),
            **eval_task_metrics,
        )
    except Exception as exc:
        log_fn(f"[eval] step {step}: runs.csv failed: {exc!r}")


__all__ = ["maybe_eval"]
