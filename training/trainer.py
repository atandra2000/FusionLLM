# training/trainer.py
"""Pre-training loop (Frozen v1 spec).

Orchestrates:
  - Model forward/backward
  - Gradient checkpointing
  - NorMuon + CautiousAdamW optimizers
  - WSD scheduler
  - MoE gate bias update
  - MTP loss computation
  - W&B logging
  - Checkpoint save/load
  - Validation

Frozen v1 spec per FINAL_FROZEN_SPEC.md §2:
  - micro_batch_size: 2
  - gradient_accumulation_steps: 16
  - total_steps: 63,400
  - dtype: bf16
  - use_checkpoint: true
  - log_interval_steps: 50
  - save_interval_steps: 2000
  - eval_interval_steps: 5000
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusionllm import FusionLLM
from models.mtp import MultiTokenPrediction
from training.optimizer import NorMuon, CautiousAdamW, build_optimizers
from training.scheduler import WSDScheduler
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from training.validation import compute_validation_loss


class Trainer:
    """Pre-training orchestrator for FusionLLM-v1.

    Single-GPU training (A100 80GB). Pure PyTorch, BF16 autocast.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.get("dtype", "bf16") == "bf16" else torch.float32

        # ── Model ─────────────────────────────────────────────────────────
        self.model = FusionLLM(config).to(self.device)
        self.use_checkpoint = config.get("use_checkpoint", True)
        self._log(f"Model built: {self._count_params():,} total params")

        # ── MTP ──────────────────────────────────────────────────────────
        mtp_depth = config.get("mtp_depth", 2)
        if mtp_depth > 0:
            self.mtp = MultiTokenPrediction(config, self.model).to(self.device)
            self._log(f"MTP enabled: depth={mtp_depth}")
        else:
            self.mtp = None

        # ── Optimizers ───────────────────────────────────────────────────
        train_model = self.mtp if self.mtp is not None else self.model
        self.muon_opt, self.adamw_opt = build_optimizers(
            train_model,
            adamw_lr=config.get("lr", 3e-4),
            muon_lr=config.get("muon_lr", 0.02),
            muon_momentum=config.get("muon_momentum", 0.95),
            adamw_betas=tuple(config.get("adamw_betas", [0.9, 0.95])),
            weight_decay=config.get("weight_decay", 0.1),
            cautious_wd=config.get("cautious_wd", True),
        )

        # ── Scheduler ────────────────────────────────────────────────────
        total_steps = config.get("total_steps", 63400)
        warmup_frac = config.get("wsd_warmup_frac", 0.01)
        stable_frac = config.get("wsd_stable_frac", 0.84)
        min_lr_ratio = config.get("min_lr_ratio", 0.1)

        # Scheduler for primary optimizer (AdamW)
        adamw_opt_for_sched = self.adamw_opt
        self.scheduler = WSDScheduler(
            adamw_opt_for_sched,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            stable_frac=stable_frac,
            min_lr_ratio=min_lr_ratio,
            decay=config.get("wsd_decay", "linear"),
        )

        # ── Training state ───────────────────────────────────────────────
        self.step = 0
        self.global_step = 0
        self.token_count = 0
        self.best_loss = float("inf")
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 16)
        self.micro_batch_size = config.get("micro_batch_size", 2)
        self.max_seq_len = config.get("max_seq_len", 4096)
        self.vocab_size = config.get("vocab_size", 64000)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.balance_loss_alpha = config.get("balance_loss_alpha", 1e-4)
        self.bias_update_speed = config.get("bias_update_speed", 1e-3)
        self.bias_update_every = config.get("bias_update_every", 10)
        self.save_dir = config.get("save_dir", "checkpoints/pretrain")
        self.save_interval = config.get("save_interval_steps", 2000)
        self.log_interval = config.get("log_interval_steps", 50)
        self.eval_interval = config.get("eval_interval_steps", 5000)
        self.max_keep = config.get("save_max_keep", 3)
        self.loss_spike_threshold = config.get("loss_spike_threshold", 3.0)
        self.grad_norm_threshold = config.get("grad_norm_threshold", 10.0)
        self.loss_nan_skip = config.get("loss_nan_skip", True)
        self.empty_cache_every = config.get("empty_cache_every", 100)

        # WandB
        self.wandb_enabled = config.get("wandb_enabled", True) and self.device.type == "cuda"
        self.wandb_run = None
        self._init_wandb()

        # Gradient scaler (disabled for BF16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def _log(self, msg: str) -> None:
        print(f"[trainer] {msg}")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _init_wandb(self) -> None:
        if not self.wandb_enabled:
            return
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.get("wandb_project", "fusionllm-v1"),
                config=self.config,
                tags=self.config.get("wandb_tags", ["v1-frozen", "single-gpu", "pure-pytorch"]),
                reinit=True,
            )
        except Exception as e:
            self._log(f"WandB init failed: {e}")
            self.wandb_enabled = False

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def _get_train_model(self) -> nn.Module:
        return self.mtp if self.mtp is not None else self.model

    def _get_moe_layers(self):
        """Get all MoE layers for bias update."""
        return self.model.get_moe_layers()

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step (micro batch).

        Args:
            tokens: (B, T) input token IDs.
            targets: (B, T) target token IDs.

        Returns:
            Dict of metrics.
        """
        train_model = self._get_train_model()
        train_model.train()

        tokens = tokens.to(self.device)
        targets = targets.to(self.device)

        # ── Forward pass (with optional gradient checkpointing) ──────────
        if self.use_checkpoint:
            def _forward(t, tgt):
                if self.mtp is not None:
                    main_logits, mtp_outputs = self.mtp(t)
                    main_loss = F.cross_entropy(
                        main_logits.view(-1, self.vocab_size),
                        tgt.view(-1),
                        reduction="mean",
                    )
                    mtp_loss = self.mtp.compute_mtp_loss(mtp_outputs)
                    return main_loss + mtp_loss
                else:
                    logits = self.model(t)
                    return F.cross_entropy(
                        logits.view(-1, self.vocab_size),
                        tgt.view(-1),
                        reduction="mean",
                    )

            with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.dtype == torch.bfloat16)):
                loss = torch.utils.checkpoint.checkpoint(
                    _forward, tokens, targets, use_reentrant=False
                )
        else:
            with torch.cuda.amp.autocast(dtype=self.dtype, enabled=(self.dtype == torch.bfloat16)):
                main_logits, mtp_outputs = self.mtp(tokens) if self.mtp is not None else (self.model(tokens), [])
                main_loss = F.cross_entropy(
                    main_logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    reduction="mean",
                )
                if mtp_outputs:
                    mtp_loss = self.mtp.compute_mtp_loss(mtp_outputs)
                else:
                    mtp_loss = torch.tensor(0.0, device=self.device)
                loss = main_loss + mtp_loss

        # ── MoE load balance loss ────────────────────────────────────────
        balance_loss = torch.tensor(0.0, device=self.device)
        if self.balance_loss_alpha > 0:
            for moe in self._get_moe_layers():
                balance_loss = balance_loss + moe.get_load_balance_loss()
            balance_loss = self.balance_loss_alpha * balance_loss
            loss = loss + balance_loss

        # ── Backward ──────────────────────────────────────────────────────
        self.scaler.scale(loss).backward()

        # ── Numerical health check ────────────────────────────────────────
        loss_val = loss.item()
        metrics = {
            "loss": loss_val,
            "main_loss": main_loss.item() if not isinstance(main_loss, torch.Tensor) or main_loss.dim() == 0 else main_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        if self.loss_nan_skip and (math.isnan(loss_val) or math.isinf(loss_val)):
            self._log(f"WARNING: NaN/Inf loss at step {self.step}, skipping")
            self.scaler.update()
            return metrics

        return metrics

    def optimizer_step(self) -> dict[str, float]:
        """Complete the optimizer step (grad clip + optimizer + scheduler)."""
        metrics = {}

        # ── Gradient clipping ─────────────────────────────────────────────
        if self.grad_clip > 0:
            train_model = self._get_train_model()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(), self.grad_clip
            )
            metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            if grad_norm > self.grad_norm_threshold:
                self._log(f"WARNING: Large grad norm {grad_norm:.2f} at step {self.step}")

        # ── Optimizer step ────────────────────────────────────────────────
        if self.muon_opt is not None:
            self.muon_opt.step()
        self.adamw_opt.step()
        self.scaler.update()

        # ── Zero gradients ────────────────────────────────────────────────
        train_model = self._get_train_model()
        train_model.zero_grad(set_to_none=True)

        # ── MoE gate bias update (every bias_update_every steps) ──────────
        if self.step % self.bias_update_every == 0 and self.step > 0:
            for moe in self._get_moe_layers():
                moe.update_gate_bias(speed=self.bias_update_speed)

        # ── Scheduler step ────────────────────────────────────────────────
        self.scheduler.step()
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.step += 1
        self.global_step += 1

        # ── Empty cache periodically ──────────────────────────────────────
        if self.empty_cache_every > 0 and self.step % self.empty_cache_every == 0:
            torch.cuda.empty_cache()

        return metrics

    def save(self, tag: str = "") -> None:
        """Save checkpoint."""
        train_model = self._get_train_model()
        save_checkpoint(
            model=train_model,
            muon_opt=self.muon_opt,
            adamw_opt=self.adamw_opt,
            scheduler=self.scheduler,
            step=self.step,
            token_count=self.token_count,
            best_loss=self.best_loss,
            save_dir=self.save_dir,
            max_keep=self.max_keep,
            tag=tag or f"step_{self.step}",
        )

    def load(self, load_dir: str | Path) -> int:
        """Load checkpoint and return the restored step."""
        train_model = self._get_train_model()
        meta = load_checkpoint(
            model=train_model,
            muon_opt=self.muon_opt,
            adamw_opt=self.adamw_opt,
            scheduler=self.scheduler,
            load_dir=load_dir,
            device=str(self.device),
        )
        restored_step = meta.get("step", 0)
        self.step = restored_step
        self.global_step = restored_step
        self.token_count = meta.get("token_count", 0)
        self.best_loss = meta.get("best_loss", float("inf"))
        self._log(f"Resumed from step {restored_step}")
        return restored_step

    def train_epoch(
        self,
        data_iter,
        total_steps: int | None = None,
    ) -> None:
        """Run training loop.

        Args:
            data_iter: Iterator yielding (tokens, targets) tensors.
            total_steps: Maximum steps. Defaults to config total_steps.
        """
        if total_steps is None:
            total_steps = self.config.get("total_steps", 63400)

        self._log(f"Starting training: {total_steps} steps, batch={self.grad_accum_steps}×{self.micro_batch_size}")
        self._log(f"Effective batch size: {self.grad_accum_steps * self.micro_batch_size} seqs")
        self._log(f"Tokens per step: {self.grad_accum_steps * self.micro_batch_size * self.max_seq_len}")

        train_model = self._get_train_model()
        t_start = time.time()

        while self.step < total_steps:
            # ── Gradient accumulation loop ────────────────────────────────
            accum_loss = 0.0

            for micro_step in range(self.grad_accum_steps):
                tokens, targets = next(data_iter)
                if tokens.size(0) != self.micro_batch_size:
                    # Skip incomplete batch
                    continue
                metrics = self.train_step(tokens, targets)

                # Scale loss for accumulation
                loss = metrics["loss"] / self.grad_accum_steps
                accum_loss += loss

            # ── Optimizer step (after accumulation) ───────────────────────
            opt_metrics = self.optimizer_step()
            self.token_count += self.grad_accum_steps * self.micro_batch_size * self.max_seq_len

            # ── Logging ──────────────────────────────────────────────────
            if self.step % self.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
                log_metrics = {
                    "loss": accum_loss / self.grad_accum_steps,
                    "step": self.step,
                    "lr": opt_metrics.get("lr", 0),
                    "grad_norm": opt_metrics.get("grad_norm", 0),
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_total": self.token_count,
                    "elapsed_sec": elapsed,
                }
                log_str = (f"step={self.step}/{total_steps} "
                           f"loss={log_metrics['loss']:.4f} "
                           f"lr={log_metrics['lr']:.2e} "
                           f"grad_norm={log_metrics['grad_norm']:.4f} "
                           f"tok/s={tokens_per_sec:.0f}")
                self._log(log_str)
                self._log_metrics(log_metrics, self.step)

            # ── Validation ────────────────────────────────────────────────
            if self.eval_interval > 0 and self.step % self.eval_interval == 0:
                val_metrics = compute_validation_loss(
                    self.model,
                    batch_size=self.micro_batch_size,
                    seq_len=self.max_seq_len,
                    vocab_size=self.vocab_size,
                    num_batches=8,
                    device=self.device,
                )
                val_metrics["step"] = self.step
                self._log(f"Validation: loss={val_metrics['loss']:.4f}, ppl={val_metrics['ppl']:.2f}")
                self._log_metrics(val_metrics, self.step)
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.save(tag="best")

            # ── Checkpoint ────────────────────────────────────────────────
            if self.save_interval > 0 and self.step % self.save_interval == 0:
                self.save()

        # ── Final save ────────────────────────────────────────────────────
        self.save(tag="final")
        self._log(f"Training complete. {total_steps} steps, {self.token_count:,} tokens")

        if self.wandb_run is not None:
            self.wandb_run.finish()
