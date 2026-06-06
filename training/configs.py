# training/configs.py
"""Configuration dataclasses for the training pipeline.

All training configurations are defined here as dataclasses.
These are used by the Pretrainer and related components.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_path: str = "data/pretrain_data.bin"
    shard_manifest_path: str | None = None
    vocab_size: int = 152064
    max_seq_len: int = 4096
    batch_size: int = 2
    gradient_accumulation_steps: int = 16


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    optimizer: str = "normuon_adamw"
    lr: float = 3e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    cautious_wd: bool = True
    max_grad_norm: float = 1.0
    scheduler: str = "wsd"
    wsd_warmup_frac: float = 0.01
    wsd_stable_frac: float = 0.84
    wsd_decay: str = "linear"


@dataclass
class ScheduleConfig:
    """Training schedule configuration."""
    max_steps: int = 50_000
    warmup_steps: int = 500
    batch_size_schedule_enabled: bool = False
    initial_batch_size: int = 2
    final_batch_size: int = 8
    batch_size_schedule_steps: int = 5_000
    seq_len_schedule_enabled: bool = False
    initial_seq_len: int = 2048
    final_seq_len: int = 8192
    seq_len_schedule_steps: int = 5_000


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    eval_enabled: bool = False
    eval_interval: int = 1_000
    eval_max_batches: int | None = 8
    eval_synthetic: bool = True
    eval_tasks: list[str] = field(default_factory=lambda: [
        "hellaswag", "arc_challenge", "piqa", "winogrande", "boolq",
    ])


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    checkpoint_dir: str = "checkpoints/pretrain"
    save_every: int = 1_000
    checkpoint_backend: str = "safetensors"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_every: int = 100
    wandb_enabled: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_tags: dict | None = None


@dataclass
class ConfigBundle:
    """Composite configuration accepted by :class:`Pretrainer`."""
    data: DataConfig = field(default_factory=DataConfig)
    model: dict = field(default_factory=dict)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dtype: str = "bf16"
    use_checkpoint: bool = True
    fuse_ce_softcap: bool = True
    fuse_linear_relu2: bool = True
    balance_loss_alpha: float = 1e-4
    z_loss_weight: float = 0.001
    bias_update_speed: float = 1e-3
    bias_update_every: int = 10
    mtp_depth: int = 3
    mtp_loss_weight: float = 0.3
    fsdp_shard_strategy: str = "FULL_SHARD"
    fsdp_forward_prefetch: bool = False
    fsdp_backward_prefetch: bool = True
    fsdp_limit_all_gathers: bool = True
    fsdp_param_dtype: str = "bf16"
    fsdp_reduce_dtype: str = "fp32"
    shard_keep_last: int = 1
    curriculum_switch_step: int = 0
    curriculum_stage1_weights: dict[str, float] | None = None
    curriculum_stage2_weights: dict[str, float] | None = None


__all__ = [
    "DataConfig",
    "OptimConfig",
    "ScheduleConfig",
    "EvalConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "ConfigBundle",
]
