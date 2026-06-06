# training/trainer.py
"""Pretrainer orchestration layer.

This module provides the Pretrainer class that orchestrates the training
loop using the decomposed modules. It delegates to:
- training.configs for configuration
- training.optimization for optimizer setup
- training.train_step for forward/backward/optimizer step
- training.validation for evaluation
- training.checkpointing for save/load
- training.curriculum_manager for curriculum learning
- training.monitoring for health monitoring
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data.async_loader import AsyncShardLoader
from models.mtp import MultiTokenPrediction
from models.transformer import Transformer, count_parameters
from training.configs import ConfigBundle
from training.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.curriculum_manager import advance_curriculum, init_curriculum
from training.dataset import PretrainDataset
from training.numerical_health import init_health_monitor, init_runs_csv, register_spike_callback
from training.optimization import Muon, build_optimizers
from training.schedules import BatchSizeSchedule, SeqLenSchedule
from training.train_step import train_step
from training.validation import maybe_eval
from training.wsd import WSDScheduler
from utils.checkpoint import CheckpointManager
from utils.distributed import (
    cleanup_distributed,
    configure_reshard,
    is_main_process,
    setup_distributed,
    wrap_fsdp2,
)
from utils.logging import get_logger, init_logging


class Pretrainer:
    """FSDP2-aware pre-training loop (v2 — accepts :class:`ConfigBundle`)."""

    def __init__(self, config: ConfigBundle):
        self.cfg = config
        self.world_size, self.rank, self.local_rank = setup_distributed()

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        log_cfg = config.logging
        data_cfg = config.data
        init_logging(
            self.rank,
            self.world_size,
            log_interval=log_cfg.log_every,
            seq_len=data_cfg.max_seq_len,
            wandb_project=log_cfg.wandb_project,
            wandb_entity=log_cfg.wandb_entity,
            wandb_run_name=log_cfg.wandb_run_name,
            wandb_tags=log_cfg.wandb_tags,
            wandb_config=asdict(config),
            wandb_enabled=log_cfg.wandb_enabled,
            mlflow_tracking_uri=log_cfg.mlflow_tracking_uri,
            mlflow_experiment_name=log_cfg.mlflow_experiment_name,
            mlflow_run_name=log_cfg.mlflow_run_name,
            mlflow_tags=log_cfg.mlflow_tags,
            mlflow_config=asdict(config),
            mlflow_enabled=log_cfg.mlflow_enabled,
        )
        self.logger = get_logger()

        self._log(f"Rank {self.rank}: initialising model...")
        raw_model = Transformer(
            config.model,
            world_size=self.world_size,
            rank=self.rank,
            use_checkpoint=config.use_checkpoint,
        ).to(self.device)

        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")

        # Optional MTP wrapper (shares main_model.embed and main_model.head).
        if getattr(raw_model, "embed", None) is not None and config.mtp_depth > 0:
            raw_model_config = dict(config.model)
            raw_model_config["mtp_depth"] = config.mtp_depth
            raw_model_config["mtp_loss_weight"] = config.mtp_loss_weight
            self.mtp = MultiTokenPrediction(raw_model_config, main_model=raw_model).to(self.device)
            self._log(f"MTP enabled: depth={config.mtp_depth}, weight={config.mtp_loss_weight}")
        else:
            self.mtp = None

        # ── FSDP2 wrap ──────────────────────────────────────────────────
        _param_dtype_map: dict[str, torch.dtype] = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        param_dtype = _param_dtype_map.get(config.fsdp_param_dtype, torch.bfloat16)
        reduce_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(config.fsdp_reduce_dtype, torch.float32)

        wrapped = wrap_fsdp2(
            self.mtp if self.mtp is not None else raw_model,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            fsdp_shard_strategy=config.fsdp_shard_strategy,
            fsdp_forward_prefetch=config.fsdp_forward_prefetch,
            fsdp_backward_prefetch=config.fsdp_backward_prefetch,
            limit_all_gathers=config.fsdp_limit_all_gathers,
        )
        if self.mtp is not None:
            self.mtp = wrapped
        else:
            self.raw_model = raw_model
        self.model = wrapped

        if self.mtp is not None and not hasattr(self, "raw_model"):
            self.raw_model = self.mtp.main_model

        # ── Optimizers (Muon/NorMuon + CautiousAdamW) ────────────────────
        self.muon, self.adamw = build_optimizers(self.raw_model, config)

        schedule_cfg = config.schedule
        optim_cfg = config.optim
        if optim_cfg.scheduler == "wsd":
            self.scheduler = WSDScheduler(
                self.adamw,
                total_steps=schedule_cfg.max_steps,
                warmup_frac=optim_cfg.wsd_warmup_frac,
                stable_frac=optim_cfg.wsd_stable_frac,
                min_lr_ratio=optim_cfg.min_lr_ratio,
                decay=optim_cfg.wsd_decay,
            )
        else:
            from training.optimization import WarmupCosineDecayScheduler
            self.scheduler = WarmupCosineDecayScheduler(
                self.adamw,
                warmup_steps=schedule_cfg.warmup_steps,
                total_steps=schedule_cfg.max_steps,
                min_lr_ratio=optim_cfg.min_lr_ratio,
            )

        self.batch_size_schedule = (
            BatchSizeSchedule(
                initial_batch_size=schedule_cfg.initial_batch_size,
                final_batch_size=schedule_cfg.final_batch_size,
                schedule_steps=schedule_cfg.batch_size_schedule_steps,
            )
            if schedule_cfg.batch_size_schedule_enabled
            else None
        )
        self.seq_len_schedule = (
            SeqLenSchedule(
                initial_seq_len=schedule_cfg.initial_seq_len,
                final_seq_len=schedule_cfg.final_seq_len,
                schedule_steps=schedule_cfg.seq_len_schedule_steps,
            )
            if schedule_cfg.seq_len_schedule_enabled
            else None
        )

        self.amp_dtype = (
            torch.bfloat16
            if config.dtype == "bf16"
            else torch.float16
            if config.dtype == "fp16"
            else None
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        configure_reshard(self.model, keep_last_n=config.shard_keep_last)

        ckpt_cfg = config.checkpoint
        self.ckpt_manager = CheckpointManager(
            ckpt_cfg.checkpoint_dir,
            checkpoint_backend=ckpt_cfg.checkpoint_backend,
        )
        self._opt_steps: int = 0
        self._last_grad_norm: float = 0.0

        # ── Curriculum (Phase 6.3) ─────────────────────────────────────
        self.curriculum = init_curriculum(config)

        # ── runs.csv logger (Phase 6.2) ────────────────────────────────
        self._runs_csv = init_runs_csv()

        # ── Numerical health monitor ──────────────────────────────────
        self.health_monitor = init_health_monitor(config)
        register_spike_callback(
            self.health_monitor,
            save_fn=self.save_checkpoint,
            log_fn=self._log,
        )

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            print(msg)

    def _amp_context(self):
        if self.amp_dtype is not None:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _maybe_eval(self, step: int) -> None:
        maybe_eval(
            step,
            self.cfg.eval,
            self.raw_model,
            self.device,
            self.logger,
            self._runs_csv,
            self._log,
        )

    def _update_schedules(self, step: int) -> None:
        data_cfg = self.cfg.data
        loader = getattr(self, "_loader", None)
        if self.batch_size_schedule is not None:
            target_bs = self.batch_size_schedule.get_batch_size(step)
            if target_bs != data_cfg.batch_size:
                self._log(f"[sched] batch_size {data_cfg.batch_size} -> {target_bs}")
                data_cfg.batch_size = target_bs
                if loader is not None and hasattr(loader, "set_batch_size"):
                    loader.set_batch_size(target_bs)
        if self.seq_len_schedule is not None:
            target_sl = self.seq_len_schedule.get_seq_len(step)
            if target_sl != data_cfg.max_seq_len:
                self._log(f"[sched] max_seq_len {data_cfg.max_seq_len} -> {target_sl}")
                data_cfg.max_seq_len = target_sl
                if loader is not None and hasattr(loader, "set_seq_len"):
                    loader.set_seq_len(target_sl)

    def train_step(
        self, tokens: torch.Tensor, targets: torch.Tensor, micro_step: int
    ) -> dict[str, float]:
        return train_step(
            model=self.model,
            mtp=self.mtp,
            raw_model=self.raw_model,
            tokens=tokens,
            targets=targets,
            micro_step=micro_step,
            cfg=self.cfg,
            muon=self.muon,
            adamw=self.adamw,
            scaler=self.scaler,
            scheduler=self.scheduler,
            health_monitor=self.health_monitor,
            device=self.device,
            rank=self.rank,
            amp_context=self._amp_context(),
        )

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        save_checkpoint(
            step=step,
            tag=tag,
            cfg=self.cfg,
            ckpt_manager=self.ckpt_manager,
            model=self.model,
            raw_model=self.raw_model,
            adamw=self.adamw,
            muon=self.muon,
            scheduler=self.scheduler,
            curriculum=self.curriculum,
            opt_steps=self._opt_steps,
            world_size=self.world_size,
            rank=self.rank,
            log_fn=self._log,
        )

    def load_checkpoint(self, step: int) -> int:
        resumed_step, opt_steps = load_checkpoint(
            step=step,
            cfg=self.cfg,
            ckpt_manager=self.ckpt_manager,
            model=self.model,
            raw_model=self.raw_model,
            adamw=self.adamw,
            muon=self.muon,
            scheduler=self.scheduler,
            curriculum=self.curriculum,
            device=str(self.device),
            log_fn=self._log,
        )
        self._opt_steps = opt_steps
        return resumed_step

    def train(self) -> None:
        data_cfg = self.cfg.data
        schedule_cfg = self.cfg.schedule
        log_cfg = self.cfg.logging

        iterate_loader: Iterator[tuple[torch.Tensor, torch.Tensor]]
        if data_cfg.shard_manifest_path and os.path.isdir(
            os.path.dirname(data_cfg.shard_manifest_path)
        ):
            loader = AsyncShardLoader(
                manifest_path=Path(data_cfg.shard_manifest_path),
                batch_size=data_cfg.batch_size,
                grad_accum=data_cfg.gradient_accumulation_steps,
                seqlen=data_cfg.max_seq_len,
                rank=self.rank,
                world_size=self.world_size,
                micro_prefetch=8,
                async_mode=torch.cuda.is_available(),
            )
            loader.start()
            iterate_loader = loader
        else:
            dataset = PretrainDataset(
                data_cfg.data_path,
                data_cfg.max_seq_len,
                data_cfg.vocab_size,
            )

            if self.world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                )
            else:
                sampler = None

            class _InfiniteIter:
                def __init__(self, dl):
                    self._dl = dl
                    self._it = iter(dl)

                def __iter__(self):
                    return self

                def __next__(self):
                    try:
                        return next(self._it)
                    except StopIteration:
                        self._it = iter(self._dl)
                        return next(self._it)

            loader = DataLoader(
                dataset,
                batch_size=data_cfg.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            iterate_loader = _InfiniteIter(loader)

        global_step = 0
        latest = find_latest_checkpoint(self.ckpt_manager)
        if latest is not None:
            try:
                global_step = self.load_checkpoint(latest)
            except Exception as exc:
                self._log(f"[warn] Could not load checkpoint: {exc}")

        self._log(
            f"Training from step {global_step} to {schedule_cfg.max_steps} "
            f"(world_size={self.world_size})"
        )
        self.model.train()

        self._loader = loader

        pbar = tqdm(total=schedule_cfg.max_steps, disable=self.rank != 0, desc="train")
        while global_step < schedule_cfg.max_steps:
            advance_curriculum(self.curriculum, global_step, loader, self._log)

            self._update_schedules(global_step)
            tokens, targets = next(iterate_loader)
            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            metrics = self.train_step(tokens, targets, global_step)

            if global_step % log_cfg.log_every == 0 and self.rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                muon_lr = self.muon.param_groups[0]["lr"] if self.muon is not None else 0.0
                self.logger.log(
                    global_step,
                    metrics["loss"],
                    lr=lr,
                    muon_lr=muon_lr,
                    grad_norm=self._last_grad_norm,
                    metrics={
                        "balance_loss": metrics["balance_loss"],
                        "z_loss": metrics["z_loss"],
                    },
                )
                if global_step % 200 == 0:
                    for i, moe in enumerate(self.raw_model.moe_layers()):
                        stats = moe.get_routing_stats()
                        if stats:
                            self.logger.log_moe_routing(global_step, i, stats)
            if global_step % self.cfg.checkpoint.save_every == 0 and global_step > 0:
                self.save_checkpoint(global_step)
            self._maybe_eval(global_step)
            global_step += 1
            pbar.update(1)

        pbar.close()
        self.save_checkpoint(global_step, tag="final")
        self._log("Training complete.")
        self.logger.finish()


__all__ = ["Pretrainer"]
