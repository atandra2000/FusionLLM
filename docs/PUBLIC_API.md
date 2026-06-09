# Public API Reference — FusionLLM

## Overview

This document defines the public API for the FusionLLM codebase. The goal is to provide clear, stable interfaces that external code can depend on, while keeping internal implementation details private.

## Design Principles

1. **Stability:** Public APIs should not change without deprecation
2. **Simplicity:** Minimize the number of public symbols
3. **Consistency:** Follow naming conventions
4. **Documentation:** Every public symbol must have a docstring

## Public API Structure

### Package Hierarchy

```
fusionllm/
├── training/
│   ├── __init__.py          # Public training API
│   ├── pretrain.py          # Entrypoint (main)
│   ├── configs.py           # Configuration classes
│   ├── trainer.py           # Pretrainer class
│   └── ...
│
├── utils/
│   ├── __init__.py          # Public utils API
│   ├── checkpoint/          # Checkpoint management
│   │   ├── __init__.py      # Re-exports CheckpointManager
│   │   └── ...
│   └── ...
│
└── models/
    ├── __init__.py          # Public models API
    ├── transformer.py       # Transformer class
    ├── moe/                 # MoE implementation
    │   ├── __init__.py      # Re-exports DeepSeekMoE
    │   └── ...
    └── ...
```

## Public API Reference

### `training` Package

#### `training/__init__.py`

```python
# Public exports
from .configs import (
    ConfigBundle,
    DataConfig,
    OptimConfig,
    ScheduleConfig,
    EvalConfig,
    CheckpointConfig,
    LoggingConfig,
)

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
]
```

#### `training/configs.py`

```python
@dataclass
class ConfigBundle:
    """Composite configuration for Pretrainer."""
    data: DataConfig
    model: dict
    optim: OptimConfig
    schedule: ScheduleConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
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
```

#### `training/trainer.py`

```python
class Pretrainer:
    """FSDP2-aware pre-training loop."""
    
    def __init__(self, config: ConfigBundle):
        """
        Initialize the pretrainer.
        
        Args:
            config: Training configuration
        """
        ...
    
    def train(self) -> None:
        """Run the full training loop."""
        ...
    
    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        micro_step: int,
    ) -> dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            tokens: Input token tensor
            targets: Target token tensor
            micro_step: Current micro step
            
        Returns:
            Dictionary of metrics
        """
        ...
    
    def save_checkpoint(self, step: int, tag: str = "") -> None:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            tag: Optional checkpoint tag
        """
        ...
    
    def load_checkpoint(self, step: int) -> int:
        """
        Load a checkpoint.
        
        Args:
            step: Checkpoint step to load
            
        Returns:
            The loaded step number
        """
        ...
```

#### `training/pretrain.py`

```python
def build_config_from_yaml(
    yaml_cfg: dict,
    args: argparse.Namespace,
) -> ConfigBundle:
    """
    Build ConfigBundle from YAML configuration.
    
    Args:
        yaml_cfg: Parsed YAML configuration
        args: CLI arguments
        
    Returns:
        ConfigBundle instance
    """
    ...

def main() -> None:
    """Main entrypoint for training."""
    ...
```

---

### `utils` Package

#### `utils/__init__.py`

```python
from .checkpoint import CheckpointManager

__all__ = [
    "CheckpointManager",
]
```

#### `utils/checkpoint/__init__.py`

```python
from .manager import CheckpointManager

__all__ = [
    "CheckpointManager",
]
```

#### `utils/checkpoint/manager.py`

```python
class CheckpointManager:
    """
    Save and load model checkpoints.
    
    Features:
        - Atomic writes via temp-file + rename
        - Sharded saves for multi-GPU
        - Optional gzip compression
        - Best validation loss tracking
        - Async checkpointing support
        - keep_last_n() pruning
    
    Usage:
        ckpt = CheckpointManager("checkpoints/pretrain")
        ckpt.save(model, optimizer, step=1000)
        meta = ckpt.load(model, step=1000, device="cuda:0")
    """
    
    def __init__(
        self,
        save_dir: str,
        async_mode: bool = True,
        sharded: bool = False,
        world_size: int = 1,
        rank: int = 0,
        compression: str = "none",
        checkpoint_backend: str = "safetensors",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory for checkpoints
            async_mode: Enable async checkpointing
            sharded: Enable sharded saves
            world_size: Number of distributed ranks
            rank: This rank's index
            compression: Compression algorithm
            checkpoint_backend: "safetensors" or "dcp"
        """
        ...
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> None:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            ema_state: Optional EMA state dict
            val_loss: Optional validation loss
            extra_meta: Optional metadata
            keep_last_n: Keep only last N checkpoints
            extra_optimizers: Additional optimizers
        """
        ...
    
    def load(
        self,
        model: torch.nn.Module,
        step: int,
        device: str = "cuda",
        optimizer: torch.optim.Optimizer | None = None,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
        strict: bool = True,
    ) -> dict:
        """
        Load checkpoint.
        
        Args:
            model: Model to load into
            step: Checkpoint step to load
            device: Device to load to
            optimizer: Optional optimizer to restore
            extra_optimizers: Additional optimizers
            strict: Strict loading
            
        Returns:
            Metadata dictionary
        """
        ...
    
    def latest_step(self) -> int | None:
        """
        Get latest complete checkpoint step.
        
        Returns:
            Latest step number, or None
        """
        ...
    
    def keep_last_n(self, n: int) -> None:
        """
        Keep only last N checkpoints.
        
        Args:
            n: Number of checkpoints to keep
        """
        ...
    
    def delete_checkpoint(self, step: int) -> None:
        """
        Delete a checkpoint.
        
        Args:
            step: Checkpoint step to delete
        """
        ...
```

#### `utils/distributed.py`

```python
def setup_distributed() -> tuple[int, int, int]:
    """
    Setup distributed training.
    
    Returns:
        (world_size, rank, local_rank)
    """
    ...

def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    ...

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce mean across ranks.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Reduced tensor
    """
    ...

def is_main_process() -> bool:
    """Check if this is the main process."""
    ...

def wrap_fsdp2(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    fsdp_shard_strategy: str = "FULL_SHARD",
    fsdp_forward_prefetch: bool = False,
    fsdp_backward_prefetch: bool = True,
    limit_all_gathers: bool = True,
) -> torch.nn.Module:
    """
    Wrap model with FSDP2.
    
    Args:
        model: Model to wrap
        param_dtype: Parameter dtype
        reduce_dtype: Reduction dtype
        fsdp_shard_strategy: Sharding strategy
        fsdp_forward_prefetch: Enable forward prefetch
        fsdp_backward_prefetch: Enable backward prefetch
        limit_all_gathers: Limit all-gather operations
        
    Returns:
        Wrapped model
    """
    ...

def configure_reshard(
    model: torch.nn.Module,
    keep_last_n: int = 1,
) -> None:
    """
    Configure per-layer reshard tuning.
    
    Args:
        model: Model to configure
        keep_last_n: Number of layers to keep
    """
    ...
```

#### `utils/logging.py`

```python
def init_logging(
    rank: int,
    world_size: int,
    log_interval: int = 100,
    seq_len: int = 4096,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_config: dict | None = None,
    wandb_enabled: bool = True,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: dict | None = None,
    mlflow_config: dict | None = None,
    mlflow_enabled: bool = True,
) -> None:
    """
    Initialize logging.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        log_interval: Logging interval
        seq_len: Sequence length
        wandb_*: W&B configuration
        mlflow_*: MLflow configuration
    """
    ...

def get_logger() -> "TrainerLogger":
    """
    Get the trainer logger.
    
    Returns:
        TrainerLogger instance
    """
    ...

class TrainerLogger:
    """Dual W&B + MLflow logger."""
    
    def log(
        self,
        step: int,
        loss: float,
        lr: float = 0.0,
        muon_lr: float = 0.0,
        grad_norm: float = 0.0,
        metrics: dict | None = None,
    ) -> None:
        """
        Log training metrics.
        
        Args:
            step: Current step
            loss: Training loss
            lr: Learning rate
            muon_lr: Muon learning rate
            grad_norm: Gradient norm
            metrics: Additional metrics
        """
        ...
    
    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_metrics: dict | None = None,
    ) -> None:
        """
        Log validation metrics.
        
        Args:
            step: Current step
            val_loss: Validation loss
            val_metrics: Additional metrics
        """
        ...
    
    def log_moe_routing(
        self,
        step: int,
        layer_idx: int,
        stats: dict,
    ) -> None:
        """
        Log MoE routing statistics.
        
        Args:
            step: Current step
            layer_idx: MoE layer index
            stats: Routing statistics
        """
        ...
    
    def finish(self) -> None:
        """Finish logging."""
        ...

class RunsCsvLogger:
    """CSV logger for experiment tracking."""
    
    def log(self, step: int, **kwargs) -> None:
        """Log metrics to CSV."""
        ...
```

#### `utils/tensor_checks.py`

```python
def validate_scalar(
    value: float,
    name: str,
    step: int | None = None,
) -> None:
    """
    Validate a scalar value is not NaN or Inf.
    
    Args:
        value: Value to validate
        name: Name for error message
        step: Optional step for error message
        
    Raises:
        RuntimeError: If value is NaN or Inf
    """
    ...

def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    step: int | None = None,
) -> None:
    """
    Validate a tensor has no NaN or Inf values.
    
    Args:
        tensor: Tensor to validate
        name: Name for error message
        step: Optional step for error message
        
    Raises:
        RuntimeError: If tensor contains NaN or Inf
    """
    ...

def validate_gradients(
    model: torch.nn.Module,
    step: int | None = None,
) -> None:
    """
    Validate all model gradients.
    
    Args:
        model: Model with gradients
        step: Optional step for error message
        
    Raises:
        RuntimeError: If any gradient is NaN or Inf
    """
    ...

def validate_loss(
    loss: torch.Tensor,
    step: int | None = None,
) -> None:
    """
    Validate loss value.
    
    Args:
        loss: Loss tensor
        step: Optional step for error message
        
    Raises:
        RuntimeError: If loss is NaN or Inf
    """
    ...
```

---

### `models` Package

#### `models/__init__.py`

```python
from .transformer import Transformer, count_parameters

__all__ = [
    "Transformer",
    "count_parameters",
]
```

#### `models/transformer.py`

```python
class Transformer(nn.Module):
    """Transformer backbone with hybrid Mamba-2 / MLA schedule."""
    
    def __init__(
        self,
        config: dict,
        world_size: int = 1,
        rank: int = 0,
        use_checkpoint: bool = True,
    ):
        """
        Initialize transformer.
        
        Args:
            config: Model configuration dict
            world_size: Number of distributed ranks
            rank: This rank's index
            use_checkpoint: Enable gradient checkpointing
        """
        ...
    
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tokens: Input token tensor
            start_pos: Start position for KV cache
            use_cache: Enable KV cache
            
        Returns:
            Logits tensor
        """
        ...
    
    def moe_layers(self) -> list:
        """Get list of MoE layers."""
        ...

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to count
        
    Returns:
        (total_params, trainable_params)
    """
    ...
```

#### `models/moe/__init__.py`

```python
from .moe import DeepSeekMoE

__all__ = [
    "DeepSeekMoE",
]
```

#### `models/moe/moe.py`

```python
class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with shared experts and aux-loss-free load balancing.
    
    Features:
        - Expert parallelism
        - Aux-loss-free load balancing
        - Expert dropout
        - Routing cache
        - Grouped-GEMM fast-path
        - All-to-all dispatch (Phase 4)
    """
    
    def __init__(
        self,
        config: dict,
        world_size: int = 1,
        rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        """
        Initialize DeepSeekMoE.
        
        Args:
            config: MoE configuration dict
            world_size: Number of distributed ranks
            rank: This rank's index
            tp_size: Tensor parallelism size
            tp_rank: Tensor parallelism rank
        """
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (T, dim)
            
        Returns:
            Output tensor (T, dim)
        """
        ...
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Get load balancing loss."""
        ...
    
    def get_z_loss(self) -> torch.Tensor:
        """Get router z-loss."""
        ...
    
    def get_routing_stats(self) -> dict[str, torch.Tensor]:
        """Get routing statistics."""
        ...
    
    def update_gate_bias(self, speed: float = 0.001) -> None:
        """
        Update gate bias for load balancing.
        
        Args:
            speed: Update speed
        """
        ...
```

---

## Internal APIs (Not Public)

These modules are internal implementation details and may change without notice:

### `training/optimization.py`
- `Muon` class
- `CautiousAdamW` class
- `build_optimizers()` function

### `training/train_step.py`
- `train_step()` function
- `optimizer_step()` function

### `training/validation.py`
- `maybe_eval()` function

### `training/checkpointing.py`
- `save_checkpoint()` function
- `load_checkpoint()` function

### `training/curriculum_manager.py`
- `init_curriculum()` function
- `advance_curriculum()` function

### `training/monitoring.py`
- `init_health_monitor()` function
- `log_metrics()` function

### `utils/checkpoint/atomic.py`
- `_atomic_save_safetensors()` function
- `_atomic_save_torch()` function
- `_atomic_save_json()` function

### `utils/checkpoint/metadata.py`
- `build_meta()` function
- `load_best_val_loss()` function
- `maybe_update_best()` function

### `utils/checkpoint/retention.py`
- `keep_last_n()` function
- `delete_checkpoint()` function
- `list_checkpoints()` function

### `utils/checkpoint/fsdp.py`
- `save_fsdp2_dcp()` function
- `execute_save_fsdp2_dcp()` function

### `utils/checkpoint/dcp.py`
- `load_fsdp2_dcp()` function

### `utils/checkpoint/recovery.py`
- `load()` function
- `load_weights()` function

### `utils/checkpoint/async_worker.py`
- `AsyncCheckpointWorker` class

### `models/moe/router.py`
- `AuxLossFreeGate` class

### `models/moe/experts.py`
- `Expert` class

### `models/moe/balancing.py`
- `get_load_balance_loss()` function
- `get_z_loss()` function

### `models/moe/dispatch.py`
- `dispatch_tokens()` function
- `compute_routing_segments()` function

### `models/moe/monitoring.py`
- `get_routing_stats()` function

### `models/moe/weight_stacks.py`
- `refresh_weight_stacks()` function

---

## Deprecation Policy

### Stable APIs (training, utils, models)

- **Major version:** Breaking changes allowed
- **Minor version:** New features, no breaking changes
- **Patch version:** Bug fixes only

### Internal APIs (optimization, train_step, etc.)

- **No guarantees:** May change at any time
- **Migration guides:** Provided for major refactors

### Deprecated APIs

```python
# Example deprecation
@deprecated("Use new_function() instead", DeprecationWarning)
def old_function():
    ...
```

---

## Usage Examples

### Basic Training

```python
from training import ConfigBundle, Pretrainer
from training.configs import DataConfig, OptimConfig

# Create config
config = ConfigBundle(
    data=DataConfig(data_path="data/train.bin"),
    model={"vocab_size": 152064, "n_layers": 24},
    optim=OptimConfig(lr=3e-4),
)

# Create trainer and run
trainer = Pretrainer(config)
trainer.train()
```

### Checkpoint Management

```python
from utils import CheckpointManager

# Create manager
ckpt = CheckpointManager("checkpoints/pretrain")

# Save checkpoint
ckpt.save(model, optimizer, step=1000)

# Load checkpoint
meta = ckpt.load(model, step=1000, device="cuda:0")

# Get latest step
latest = ckpt.latest_step()
```

### Distributed Training

```python
from utils.distributed import setup_distributed, wrap_fsdp2

# Setup
world_size, rank, local_rank = setup_distributed()

# Wrap model
model = wrap_fsdp2(model, param_dtype=torch.bfloat16)
```

---

*Public API reference created: Phase 3 Architectural Modularization*
