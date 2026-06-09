# utils/logging.py
"""Training metrics — W&B logging; stdout is the tertiary sink.

Identity
--------
* **W&B is the only logging backend.** 
* Async log submissions via a small thread pool keep the trainer hot
  path off the network.
* Histograms (MoE expert load) are submitted to W&B.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch


class TrainerLogger:
    """
    Logs training and validation metrics to **W&B**, with stdout as the tertiary sink.
    """

    def __init__(
        self,
        log_interval: int = 10,
        seq_len: int = 4096,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_tags: list | None = None,
        wandb_config: dict[str, Any] | None = None,
        wandb_enabled: bool = True,
        log_grad_norm: bool = True,
    ):
        self.log_interval = log_interval
        self.seq_len = seq_len
        self.log_grad_norm = log_grad_norm
        self.wandb_enabled = wandb_enabled

        self._start = time.time()
        self._step_start = time.time()
        self._step_tokens: int = 0
        self._loss_window: list[float] = []
        self._ema_loss: float | None = None
        self._ema_alpha: float = 0.1

        self._wandb = None

        self._async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="logger_async")

        # ── W&B ────────────────────────────────────────────────────────
        if wandb_enabled and wandb_project:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity or None,
                    name=wandb_run_name,
                    tags=wandb_tags or [],
                    config=wandb_config or {},
                    reinit=True,
                )
                self._wandb = wandb
                if wandb.run is not None:
                    print(f"[logging] W&B run: {wandb.run.url}")
            except ImportError:
                print("[logging] wandb not installed — pip install wandb")
                self.wandb_enabled = False
            except Exception as exc:
                print(f"[logging] W&B init failed: {exc}")
                self.wandb_enabled = False
        elif wandb_enabled and not wandb_project:
            print("[logging] wandb.project not set — W&B disabled")

    def _gpu_stats(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "system/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
            "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "system/gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def log(
        self,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        lr: float = 0.0,
        muon_lr: float = 0.0,
        batch_size: int = 1,
        grad_norm: float | None = None,
    ) -> None:
        self._loss_window.append(loss)
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = (1.0 - self._ema_alpha) * self._ema_loss + self._ema_alpha * loss

        self._step_tokens += batch_size * self.seq_len

        if step % self.log_interval != 0 or not self._loss_window:
            return

        avg_loss = sum(self._loss_window) / len(self._loss_window)
        elapsed = max(time.time() - self._step_start, 1e-6)
        tokens_per_sec = self._step_tokens / elapsed
        ppl = torch.tensor(avg_loss).exp().item()

        parts = [
            f"step={step:>7}",
            f"loss={avg_loss:.4f}",
            f"ema={self._ema_loss:.4f}",
            f"ppl={ppl:.2f}",
            f"lr={lr:.2e}",
            f"tps={tokens_per_sec:,.0f}",
        ]
        if metrics:
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.3f}")
        print(" | ".join(parts))

        log_dict: dict[str, Any] = {
            "train/loss": avg_loss,
            "train/ema_loss": self._ema_loss,
            "train/ppl": ppl,
            "train/lr_adamw": lr,
            "train/lr_muon": muon_lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/tokens_this_log": self._step_tokens,
            "train/elapsed_sec": elapsed,
        }
        if grad_norm is not None:
            log_dict["train/grad_norm"] = grad_norm
        if metrics:
            for k, v in metrics.items():
                log_dict[f"train/{k}"] = v
        log_dict.update(self._gpu_stats())

        # ── W&B ────────────────────────────────────────────────────────
        if self._wandb is not None:
            if self._async_executor:
                self._async_executor.submit(self._wandb.log, log_dict, step=step)
            else:
                self._wandb.log(log_dict, step=step)

        self._loss_window = []
        self._step_tokens = 0
        self._step_start = time.time()

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        val_ppl = torch.tensor(val_loss).exp().item()
        parts = [
            f"[VAL] step={step:>7}",
            f"val_loss={val_loss:.4f}",
            f"val_ppl={val_ppl:.2f}",
        ]
        if val_metrics:
            for k, v in val_metrics.items():
                parts.append(f"{k}={v:.4f}")
        print(" | ".join(parts))

        log_dict: dict[str, Any] = {
            "val/loss": val_loss,
            "val/ppl": val_ppl,
        }
        if val_metrics:
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
        log_dict.update(self._gpu_stats())

        if self._wandb is not None:
            self._async_executor.submit(self._wandb.log, log_dict, step=step)

    def log_moe_routing(self, step: int, layer_idx: int, stats: dict[str, torch.Tensor]) -> None:
        """Log per-expert load histograms (sparse step logging)."""
        if "load" not in stats:
            return
        load = stats["load"].detach().float().cpu().tolist()

        if self._wandb is not None:
            self._async_executor.submit(
                self._wandb.log,
                {f"moe/layer_{layer_idx}/expert_load": self._wandb.Histogram(load)},
                step=step,
            )

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a file artefact to W&B (rank-0 only)."""
        if self._wandb is not None:
            self._async_executor.submit(self._wandb.log_artifact, local_path, artifact_path)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._wandb is not None and self._wandb.run is not None:
            for k, v in summary.items():
                self._wandb.run.summary[k] = v

    def save_log(self, filename: str, data: dict) -> None:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def finish(self) -> None:
        if self._async_executor:
            self._async_executor.shutdown(wait=True)
        if self._wandb is not None:
            self._wandb.finish()


_logger: TrainerLogger | None = None


def init_logging(
    rank: int = 0,
    world_size: int = 1,
    log_interval: int = 10,
    seq_len: int = 4096,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list | None = None,
    wandb_config: dict[str, Any] | None = None,
    wandb_enabled: bool = True,
    log_grad_norm: bool = True,
) -> None:
    """Module-level initialiser (called by the trainer once)."""
    global _logger
    _ = rank, world_size
    _logger = TrainerLogger(
        log_interval=log_interval,
        seq_len=seq_len,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_config=wandb_config,
        wandb_enabled=wandb_enabled,
        log_grad_norm=log_grad_norm,
    )


def get_logger() -> TrainerLogger:
    global _logger
    if _logger is None:
        _logger = TrainerLogger()
    return _logger


# ── Runs CSV logger (Phase 6.2) ──────────────────────────────────────────────
class RunsCsvLogger:
    """Append-only CSV logger that writes eval metrics to ``runs.csv``.

    Created once per training run.  Each call to ``log`` appends a row
    with step, loss, ppl, and optional extra columns.  The file is
    created on first write with a header row.
    """

    def __init__(self, path: str = "runs.csv"):
        self.path = Path(path)
        self._header_written = self.path.exists()

    def log(self, step: int, loss: float, ppl: float, **extra: float | int | str) -> None:
        row: dict[str, float | int | str] = {
            "step": step,
            "loss": loss,
            "ppl": ppl,
        }
        row.update(extra)
        with open(self.path, "a") as f:
            if not self._header_written:
                f.write(",".join(str(k) for k in row) + "\n")
                self._header_written = True
            f.write(",".join(str(v) for v in row.values()) + "\n")
