# training/__init__.py
"""Public API for the training package."""

from .configs import (
    CheckpointConfig,
    ConfigBundle,
    DataConfig,
    EvalConfig,
    LoggingConfig,
    OptimConfig,
    ScheduleConfig,
)
from .dataset import PretrainDataset
from .optimization import (
    CautiousAdamW,
    Muon,
    WarmupCosineDecayScheduler,
    _cautious_mask,
    _zeropower_via_newtonschulz5,
    build_optimizers,
)
from .pretrain import build_config_from_yaml
from .trainer import Pretrainer

__all__ = [
    # Configuration
    "ConfigBundle",
    "DataConfig",
    "OptimConfig",
    "ScheduleConfig",
    "EvalConfig",
    "CheckpointConfig",
    "LoggingConfig",
    # Training
    "Pretrainer",
    "PretrainDataset",
    # Optimization
    "Muon",
    "CautiousAdamW",
    "WarmupCosineDecayScheduler",
    "build_optimizers",
    "_cautious_mask",
    "_zeropower_via_newtonschulz5",
    # Entrypoint
    "build_config_from_yaml",
]
