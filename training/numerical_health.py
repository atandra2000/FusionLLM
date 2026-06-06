# training/numerical_health.py
"""Numerical health checks for training stability.

Prevents silent training failures through comprehensive monitoring:
- Loss spike detection with configurable thresholds
- Gradient anomaly detection
- Activation monitoring (optional)
- Automatic checkpoint saving on anomaly detection
"""

import torch
import torch.nn as nn
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from collections import deque

from utils.tensor_checks import validate_scalar, validate_gradients as _validate_gradients


@dataclass
class HealthConfig:
    """Configuration for numerical health checks."""
    # Loss spike detection
    loss_spike_window: int = 100          # Rolling window size for loss statistics
    loss_spike_threshold: float = 5.0     # Z-score threshold for spike detection
    loss_spike_abs_threshold: float = 10.0 # Absolute delta threshold
    
    # Gradient anomaly detection
    grad_norm_window: int = 100           # Rolling window for gradient norms
    grad_norm_threshold: float = 10.0     # Z-score threshold for gradient anomaly
    grad_norm_max: float = 100.0          # Maximum allowed gradient norm
    
    # Activation monitoring
    check_activations: bool = False       # Enable activation monitoring
    activation_nan_check: bool = True     # Check for NaN in activations
    activation_inf_check: bool = True     # Check for Inf in activations
    
    # Alerting
    alert_on_spike: bool = True           # Print alert on spike detection
    save_on_spike: bool = True            # Save checkpoint on spike detection
    
    # EMA for loss tracking
    ema_decay: float = 0.99              # EMA decay for loss smoothing


class NumericalHealthMonitor:
    """Monitors numerical health during training."""
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        
        # Loss tracking
        self.loss_history: deque = deque(maxlen=self.config.loss_spike_window)
        self.loss_ema: Optional[float] = None
        self.loss_ema_var: float = 0.0
        
        # Gradient tracking
        self.grad_norm_history: deque = deque(maxlen=self.config.grad_norm_window)
        
        # Activation tracking
        self.activation_stats: Dict[str, Dict] = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # State
        self.is_active = True
        self.spike_count = 0
        self.last_spike_step = -1
        
    def register_alert_callback(self, callback: Callable):
        """Register a callback to be called on anomaly detection."""
        self.alert_callbacks.append(callback)
    
    def update_loss(self, loss: float, step: int) -> bool:
        """Update loss statistics and check for spikes.
        
        Args:
            loss: Current loss value
            step: Current training step
            
        Returns:
            True if spike detected, False otherwise
            
        Raises:
            RuntimeError: If loss is NaN or Inf
        """
        if not self.is_active:
            return False
        
        # ── NaN/Inf check — fail loudly ────────────────────────────────
        validate_scalar(loss, "loss", step)
        
        # Update EMA
        if self.loss_ema is None:
            self.loss_ema = loss
            self.loss_ema_var = 0.0
        else:
            delta = loss - self.loss_ema
            self.loss_ema = self.config.ema_decay * self.loss_ema + (1 - self.config.ema_decay) * loss
            self.loss_ema_var = self.config.ema_decay * self.loss_ema_var + (1 - self.config.ema_decay) * delta * delta
        
        # Add to history
        self.loss_history.append(loss)
        
        # Check for spike
        if len(self.loss_history) < 10:
            return False
        
        # Calculate statistics
        history_list = list(self.loss_history)
        mean = sum(history_list) / len(history_list)
        variance = sum((x - mean) ** 2 for x in history_list) / len(history_list)
        std = math.sqrt(variance) if variance > 0 else 1e-8
        
        # Z-score check
        z_score = abs(loss - mean) / std if std > 0 else 0
        
        # Absolute delta check
        abs_delta = abs(loss - self.loss_ema)
        
        # Detect spike
        is_spike = (z_score > self.config.loss_spike_threshold or 
                    abs_delta > self.config.loss_spike_abs_threshold)
        
        if is_spike:
            self.spike_count += 1
            self.last_spike_step = step
            
            if self.config.alert_on_spike:
                self._alert(
                    f"Loss spike detected at step {step}: "
                    f"loss={loss:.4f}, z_score={z_score:.2f}, "
                    f"ema_delta={abs_delta:.4f}"
                )
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(step, "loss_spike", {
                        "loss": loss,
                        "z_score": z_score,
                        "ema_delta": abs_delta,
                    })
                except Exception as e:
                    print(f"Alert callback error: {e}")
        
        return is_spike
    
    def update_gradients(self, model: nn.Module, step: int) -> bool:
        """Update gradient statistics and check for anomalies.
        
        Args:
            model: Model to check gradients for
            step: Current training step
            
        Returns:
            True if anomaly detected, False otherwise
            
        Raises:
            RuntimeError: If NaN or Inf gradients detected
        """
        if not self.is_active:
            return False
        
        # ── NaN/Inf check on gradients — fail loudly ───────────────────
        _validate_gradients(model, step=step)
        
        # Calculate total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Add to history
        self.grad_norm_history.append(total_norm)
        
        # Check for anomaly
        if len(self.grad_norm_history) < 10:
            return False
        
        # Calculate statistics
        history_list = list(self.grad_norm_history)
        mean = sum(history_list) / len(history_list)
        variance = sum((x - mean) ** 2 for x in history_list) / len(history_list)
        std = math.sqrt(variance) if variance > 0 else 1e-8
        
        # Z-score check
        z_score = abs(total_norm - mean) / std if std > 0 else 0
        
        # Absolute max check
        is_anomaly = (z_score > self.config.grad_norm_threshold or
                      total_norm > self.config.grad_norm_max)
        
        if is_anomaly:
            if self.config.alert_on_spike:
                self._alert(
                    f"Gradient anomaly detected at step {step}: "
                    f"norm={total_norm:.4f}, z_score={z_score:.2f}"
                )
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(step, "grad_anomaly", {
                        "grad_norm": total_norm,
                        "z_score": z_score,
                    })
                except Exception as e:
                    print(f"Alert callback error: {e}")
        
        return is_anomaly
    
    def check_activations(self, activations: Dict[str, torch.Tensor], step: int) -> bool:
        """Check activations for NaN or Inf.
        
        Args:
            activations: Dictionary of activation tensors
            step: Current training step
            
        Returns:
            True if anomaly detected, False otherwise
        """
        if not self.is_active or not self.config.check_activations:
            return False
        
        anomalies_found = False
        
        for name, tensor in activations.items():
            has_nan = self.config.activation_nan_check and torch.isnan(tensor).any()
            has_inf = self.config.activation_inf_check and torch.isinf(tensor).any()
            
            if has_nan or has_inf:
                anomalies_found = True
                self._alert(
                    f"Activation anomaly at step {step}: "
                    f"layer={name}, has_nan={has_nan}, has_inf={has_inf}"
                )
                
                # Update stats
                if name not in self.activation_stats:
                    self.activation_stats[name] = {"nan_count": 0, "inf_count": 0}
                
                if has_nan:
                    self.activation_stats[name]["nan_count"] += 1
                if has_inf:
                    self.activation_stats[name]["inf_count"] += 1
        
        return anomalies_found
    
    def _alert(self, message: str):
        """Print alert message."""
        print(f"\n{'='*60}")
        print(f"⚠️  NUMERICAL HEALTH ALERT")
        print(f"{'='*60}")
        print(f"{message}")
        print(f"{'='*60}\n")
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stats = {
            "spike_count": self.spike_count,
            "last_spike_step": self.last_spike_step,
            "loss_ema": self.loss_ema,
            "loss_ema_std": math.sqrt(self.loss_ema_var) if self.loss_ema_var > 0 else 0,
        }
        
        if self.loss_history:
            history_list = list(self.loss_history)
            stats["loss_mean"] = sum(history_list) / len(history_list)
            stats["loss_min"] = min(history_list)
            stats["loss_max"] = max(history_list)
        
        if self.grad_norm_history:
            history_list = list(self.grad_norm_history)
            stats["grad_norm_mean"] = sum(history_list) / len(history_list)
            stats["grad_norm_max"] = max(history_list)
        
        return stats
    
    def reset(self):
        """Reset all statistics."""
        self.loss_history.clear()
        self.loss_ema = None
        self.loss_ema_var = 0.0
        self.grad_norm_history.clear()
        self.activation_stats.clear()
        self.spike_count = 0
        self.last_spike_step = -1


class ActivationMonitor:
    """Hook-based activation monitor for nn.Module."""
    
    def __init__(self, model: nn.Module, config: Optional[HealthConfig] = None):
        self.model = model
        self.config = config or HealthConfig()
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Callable] = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all layers."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
    
    def _create_hook(self, name: str):
        """Create a hook function for a named module."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    self.activations[name] = output[0].detach()
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get current activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()


def create_health_monitor(
    config: Optional[HealthConfig] = None,
    alert_callback: Optional[Callable] = None,
) -> NumericalHealthMonitor:
    """Create a health monitor with optional alert callback.
    
    Args:
        config: Health check configuration
        alert_callback: Optional callback for alerts
        
    Returns:
        Configured NumericalHealthMonitor
    """
    monitor = NumericalHealthMonitor(config)
    if alert_callback:
        monitor.register_alert_callback(alert_callback)
    return monitor


def init_health_monitor(cfg) -> NumericalHealthMonitor:
    """Initialize the numerical health monitor from ConfigBundle.

    Args:
        cfg: ConfigBundle with logging settings

    Returns:
        Configured NumericalHealthMonitor
    """
    return NumericalHealthMonitor(
        HealthConfig(
            loss_spike_window=cfg.logging.log_every * 10,
            loss_spike_threshold=5.0,
            loss_spike_abs_threshold=10.0,
            grad_norm_threshold=10.0,
            grad_norm_max=100.0,
        )
    )


def register_spike_callback(
    health_monitor: NumericalHealthMonitor,
    save_fn: Callable[[int, str], None],
    log_fn: Callable[[str], None],
) -> None:
    """Register a callback to save checkpoint on spike detection.

    Args:
        health_monitor: NumericalHealthMonitor instance
        save_fn: Function to save checkpoint (step, tag)
        log_fn: Logging function
    """
    def _on_spike(step: int, alert_type: str, data: dict):
        if alert_type == "loss_spike":
            log_fn(f"[Health] Saving emergency checkpoint at step {step}")
            save_fn(step, f"spike_{step}")

    health_monitor.register_alert_callback(_on_spike)


def init_runs_csv():
    """Initialize the runs CSV logger."""
    from utils.logging import RunsCsvLogger
    return RunsCsvLogger()
