"""Optional lm-eval-harness wrapper for eval during training (Phase 6.2).

Gracefully handles missing ``lm_eval`` package — returns ``None`` when
the package is not installed or when running on CPU.  The caller
(``Pretrainer._maybe_eval``) is expected to swallow ``None`` results.
"""

from __future__ import annotations

from typing import Any

import torch


def run_lm_eval(
    model: torch.nn.Module,
    tasks: list[str] | None = None,
    *,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, float] | None:
    """Run a set of lm-eval-harness tasks on *model*.

    Args:
        model: a model with a ``forward(tokens, start_pos=0, use_cache=False)``
               signature (the project's ``Transformer``).
        tasks: list of task names (default: HellaSwag, ARC-c, PIQA).
        device: target device.
        limit: limit samples per task (default: None = all).

    Returns:
        A dict mapping ``{task_name: score}``, or ``None`` if the
        ``lm_eval`` package is unavailable or CUDA is not available.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return None

    try:
        import lm_eval  # noqa: F401
        from lm_eval.evaluator import simple_evaluate
    except ImportError:
        return None

    if tasks is None:
        tasks = ["hellaswag", "arc_challenge", "piqa", "winogrande", "boolq"]

    try:
        results = simple_evaluate(
            model=model,
            tasks=tasks,
            device=device,
            batch_size="auto",
            limit=limit,
            log_samples=False,
        )
    except Exception:
        return None

    if results is None or "results" not in results:
        return None

    return {task: float(results["results"][task].get("acc,none", 0.0)) for task in tasks if task in results["results"]}
