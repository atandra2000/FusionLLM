# training/pretrain.py
"""Single-process / **FSDP2** pre-training loop (v2).

Identity
--------
* **FSDP2** (``torch.distributed.fsdp.fully_shard``) on the canonical
  8×A100 SXM 80GB target.  DDP is intentionally not supported.
* **NorMuon for matrix params + Cautious AdamW for the rest** (embeddings,
  LM head, MTP projections, norms, MoE gate).
* BF16 autocast on forward, FP32 reduction in attention softmax.
* Per-expert aux-loss-free bias update via ``DeepSeekMoE.update_gate_bias``.
* Async checkpointing is the caller's choice (``CheckpointManager`` is
  already async; this loop just calls ``ckpt_manager.save()``).
* W&B logging; TrainerLogger logs to W&B.

This module is the thin entrypoint. Training logic is in trainer.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from training.configs import (
    CheckpointConfig,
    ConfigBundle,
    DataConfig,
    EvalConfig,
    LoggingConfig,
    OptimConfig,
    ScheduleConfig,
)
from training.trainer import Pretrainer
from utils.distributed import cleanup_distributed


def build_config_from_yaml(yaml_cfg: dict, args: argparse.Namespace) -> ConfigBundle:
    """Build ConfigBundle from YAML configuration and CLI arguments."""
    t = yaml_cfg.get("training", {})
    d = yaml_cfg.get("data", {})
    m = yaml_cfg.get("model", {})
    data_path = args.data_path or d.get("train_data_path", "data/pretrain_data.bin")
    shard_manifest = d.get("shard_manifest_path")

    data_cfg = DataConfig(
        data_path=data_path,
        shard_manifest_path=shard_manifest,
        vocab_size=m.get("vocab_size", 152064),
        max_seq_len=m.get("max_seq_len", 4096),
        batch_size=t.get("micro_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
    )

    optim_cfg = OptimConfig(
        optimizer=t.get("optimizer", "normuon_adamw"),
        lr=t.get("lr", 3e-4),
        muon_lr=t.get("muon_lr", 0.02),
        muon_momentum=t.get("muon_momentum", 0.95),
        adamw_betas=tuple(t.get("adamw_betas", [0.9, 0.95])),
        min_lr_ratio=t.get("min_lr_ratio", 0.1),
        weight_decay=t.get("weight_decay", 0.1),
        cautious_wd=t.get("cautious_wd", True),
        max_grad_norm=t.get("grad_clip", 1.0),
        scheduler=t.get("scheduler", "wsd"),
        wsd_warmup_frac=t.get("wsd_warmup_frac", 0.01),
        wsd_stable_frac=t.get("wsd_stable_frac", 0.84),
        wsd_decay=t.get("wsd_decay", "linear"),
    )

    schedule_cfg = ScheduleConfig(
        max_steps=t.get("total_steps", 50_000),
        warmup_steps=t.get("warmup_steps", 500),
        batch_size_schedule_enabled=t.get("batch_size_schedule_enabled", False),
        initial_batch_size=t.get("initial_batch_size", 2),
        final_batch_size=t.get("final_batch_size", 8),
        batch_size_schedule_steps=t.get("batch_size_schedule_steps", 5_000),
        seq_len_schedule_enabled=t.get("seq_len_schedule_enabled", False),
        initial_seq_len=t.get("initial_seq_len", 2048),
        final_seq_len=t.get("final_seq_len", 8192),
        seq_len_schedule_steps=t.get("seq_len_schedule_steps", 5_000),
    )

    eval_cfg = EvalConfig(
        eval_enabled=t.get("eval_enabled", False),
        eval_interval=t.get("eval_interval", 1_000),
        eval_max_batches=t.get("eval_max_batches", 8),
        eval_synthetic=t.get("eval_synthetic", True),
        eval_tasks=t.get("eval_tasks", [
            "hellaswag", "arc_challenge", "piqa", "winogrande", "boolq",
        ]),
    )

    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain"),
        save_every=t.get("save_interval", 1_000),
        checkpoint_backend=t.get("checkpoint_backend", "safetensors"),
    )

    log_cfg = LoggingConfig(
        log_every=t.get("log_interval", 100),
        wandb_enabled=t.get("wandb_enabled", True),
        wandb_project=t.get("wandb_project"),
        wandb_entity=t.get("wandb_entity"),
        wandb_run_name=t.get("wandb_run_name"),
        wandb_tags=t.get("wandb_tags", []),
    )

    return ConfigBundle(
        data=data_cfg,
        model=m,
        optim=optim_cfg,
        schedule=schedule_cfg,
        eval=eval_cfg,
        checkpoint=ckpt_cfg,
        logging=log_cfg,
        dtype=t.get("dtype", "bf16"),
        use_checkpoint=args.use_checkpoint or t.get("use_checkpoint", True),
        fuse_ce_softcap=t.get("fuse_ce_softcap", True),
        fuse_linear_relu2=t.get("fuse_linear_relu2", True),
        balance_loss_alpha=t.get("balance_loss_alpha", 1e-4),
        z_loss_weight=t.get("z_loss_weight", 0.001),
        bias_update_speed=t.get("bias_update_speed", 1e-3),
        bias_update_every=t.get("bias_update_every", 10),
        mtp_depth=m.get("mtp_depth", 3),
        mtp_loss_weight=m.get("mtp_loss_weight", 0.3),
        fsdp_shard_strategy=t.get("fsdp_shard_strategy", "FULL_SHARD"),
        fsdp_forward_prefetch=t.get("fsdp_forward_prefetch", False),
        fsdp_backward_prefetch=t.get("fsdp_backward_prefetch", True),
        fsdp_limit_all_gathers=t.get("fsdp_limit_all_gathers", True),
        fsdp_param_dtype=t.get("fsdp_param_dtype", "bf16"),
        fsdp_reduce_dtype=t.get("fsdp_reduce_dtype", "fp32"),
        shard_keep_last=t.get("shard_keep_last", 1),
        curriculum_switch_step=t.get("curriculum_switch_step", 0),
        curriculum_stage1_weights=t.get("curriculum_stage1_weights"),
        curriculum_stage2_weights=t.get("curriculum_stage2_weights"),
    )


def main() -> None:
    """Main entrypoint for training."""
    parser = argparse.ArgumentParser(description="FusionLLM pre-training (FSDP2)")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint step to resume from")
    parser.add_argument(
        "--use-checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    config = build_config_from_yaml(yaml_cfg, args)
    trainer = Pretrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))
    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()
