# Monitoring Guide

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: Monitoring and Logging Guide

---

## Executive Summary

This document outlines the monitoring strategy for FusionLLM V2, covering:
1. **Core metrics**: Loss, gradients, activation norms
2. **Router metrics**: Entropy, utilization, collapse indicators
3. **Performance metrics**: Throughput, GPU utilization, communication
4. **Dashboards**: W&B and MLflow integration

---

## 1. Core Training Metrics

### Loss Tracking

```python
# In training loop
def compute_metrics(loss, model, batch, step):
    metrics = {
        'train/loss': loss.item(),
        'train/perplexity': torch.exp(loss).item(),
        'train/learning_rate': optimizer.param_groups[0]['lr'],
        'train/step': step,
        'train/tokens_per_sec': batch['input_ids'].numel() / elapsed,
    }
    return metrics
```

### Gradient Metrics

```python
def compute_gradient_metrics(model):
    grad_norms = {}
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[f'grad_norm/{name}'] = grad_norm
            total_norm += grad_norm ** 2
    
    grad_norms['grad_norm/total'] = total_norm ** 0.5
    
    # Log gradient explosion warnings
    if total_norm ** 0.5 > 10.0:
        grad_norms['grad_norm/explosion_warning'] = 1.0
    
    return grad_norms
```

### Activation Norms

```python
def compute_activation_norms(model):
    """Capture activation norms during forward pass."""
    activation_norms = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_norms[f'activation_norm/{name}'] = output.norm().item()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    output = model(input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_norms
```

---

## 2. Router Metrics

### Routing Entropy

```python
def compute_routing_entropy(routing_stats):
    """Compute entropy of routing distribution."""
    load = routing_stats['load']
    
    # Entropy: -sum(p * log(p))
    entropy = -(load * torch.log(load + 1e-10)).sum()
    
    return {
        'router/entropy': entropy.item(),
        'router/entropy_normalized': entropy.item() / math.log(load.numel()),
    }
```

### Expert Utilization

```python
def compute_expert_utilization(routing_stats):
    """Compute expert utilization metrics."""
    counts = routing_stats['counts']
    load = routing_stats['load']
    
    # Utilization: fraction of experts receiving tokens
    utilization = (counts > 0).float().mean()
    
    # Load variance
    load_variance = load.var()
    
    # Max/min load ratio
    max_load = load.max()
    min_load = load.min()
    load_ratio = max_load / (min_load + 1e-10)
    
    return {
        'router/utilization': utilization.item(),
        'router/load_variance': load_variance.item(),
        'router/max_load': max_load.item(),
        'router/min_load': min_load.item(),
        'router/load_ratio': load_ratio.item(),
    }
```

### Expert Collapse Detection

```python
class ExpertCollapseDetector:
    def __init__(self, config):
        self.entropy_threshold = config.get('entropy_threshold', 0.5)
        self.utilization_threshold = config.get('utilization_threshold', 0.1)
        
        self.entropy_history = []
        self.utilization_history = []
    
    def update(self, routing_stats):
        entropy = compute_routing_entropy(routing_stats)
        utilization = compute_expert_utilization(routing_stats)
        
        self.entropy_history.append(entropy['router/entropy'])
        self.utilization_history.append(utilization['router/utilization'])
    
    def check_collapse(self):
        warnings = []
        
        if len(self.entropy_history) >= 100:
            avg_entropy = sum(self.entropy_history[-100:]) / 100
            if avg_entropy < self.entropy_threshold:
                warnings.append(f"Low entropy: {avg_entropy:.3f}")
        
        if len(self.utilization_history) >= 100:
            avg_utilization = sum(self.utilization_history[-100:]) / 100
            if avg_utilization < self.utilization_threshold:
                warnings.append(f"Low utilization: {avg_utilization:.3f}")
        
        return warnings
```

---

## 3. Performance Metrics

### Throughput Tracking

```python
def compute_throughput_metrics(batch, elapsed_time):
    tokens_per_batch = batch['input_ids'].numel()
    tokens_per_sec = tokens_per_batch / elapsed_time
    
    return {
        'perf/tokens_per_sec': tokens_per_sec,
        'perf/batch_time_ms': elapsed_time * 1000,
        'perf/tokens_per_step': tokens_per_batch,
    }
```

### GPU Utilization

```python
def compute_gpu_metrics():
    """Track GPU utilization and memory."""
    gpu_metrics = {}
    
    # Memory usage
    gpu_metrics['gpu/memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
    gpu_metrics['gpu/memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
    gpu_metrics['gpu/memory_max'] = torch.cuda.max_memory_allocated() / 1024**3
    
    # Utilization (requires nvidia-smi)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        gpu_metrics['gpu/utilization'] = float(result.stdout.strip()) / 100
    except:
        gpu_metrics['gpu/utilization'] = 0.0
    
    return gpu_metrics
```

### Communication Overhead

```python
def compute_communication_metrics():
    """Track communication overhead."""
    # This requires profiling NCCL operations
    # For now, return placeholder metrics
    return {
        'comm/all_reduce_time': 0.0,
        'comm/all_gather_time': 0.0,
        'comm/communication_bytes': 0.0,
    }
```

---

## 4. Dashboard Integration

### W&B Dashboard

```python
# utils/logging.py
import wandb

class WandBLogger:
    def __init__(self, project, entity=None, run_name=None):
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
        )
    
    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)
    
    def log_model(self, model, step):
        """Log model gradients and parameters."""
        wandb.log({
            'gradients': wandb.Histogram(model.grad),
            'parameters': wandb.Histogram(model.param),
        }, step=step)
    
    def log_attention(self, attention_weights, step):
        """Log attention visualization."""
        wandb.log({
            'attention': wandb.Image(attention_weights),
        }, step=step)
```

### MLflow Dashboard

```python
# utils/logging.py
import mlflow

class MLflowLogger:
    def __init__(self, tracking_uri, experiment_name, run_name=None):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
    
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_artifact(self, local_path):
        mlflow.log_artifact(local_path)
```

### Combined Logger

```python
class TrainerLogger:
    def __init__(self, config):
        self.wandb = WandBLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            run_name=config.wandb_run_name,
        ) if config.wandb_enabled else None
        
        self.mlflow = MLflowLogger(
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name,
            run_name=config.mlflow_run_name,
        ) if config.mlflow_enabled else None
    
    def log(self, metrics, step):
        if self.wandb:
            self.wandb.log_metrics(metrics, step)
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step)
```

---

## 5. Dashboard Layout

### W&B Dashboard Panels

```
┌─────────────────────────────────────────────────────────┐
│ Training Progress                                       │
├─────────────────────────────────────────────────────────┤
│ Loss: 2.345 (↓) │ Perplexity: 10.43 (↓) │ LR: 3e-4   │
├─────────────────────────────────────────────────────────┤
│ Gradient Norm: 0.823 │ Activation Norm: 12.45          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Expert Routing                                          │
├─────────────────────────────────────────────────────────┤
│ Utilization: 0.45 │ Entropy: 1.23 │ Load Ratio: 2.1    │
├─────────────────────────────────────────────────────────┤
│ [Expert Heatmap] [Load Distribution] [Entropy Timeline]│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Performance                                             │
├─────────────────────────────────────────────────────────┤
│ Tokens/sec: 3.5M │ Batch Time: 36ms │ GPU Util: 85%    │
├─────────────────────────────────────────────────────────┤
│ Memory: 14.2/80 GB │ Communication: 2.5 ms             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Stability                                               │
├─────────────────────────────────────────────────────────┤
│ NaN Count: 0 │ Inf Count: 0 │ Grad Explosion: 0        │
├─────────────────────────────────────────────────────────┤
│ EMA Decay: 0.9999 │ EMA Loss: 2.340                    │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Alerting

### Threshold Alerts

```python
class AlertManager:
    def __init__(self, config):
        self.alerts = {
            'loss_spike': {'threshold': 2.0, 'window': 100},
            'gradient_explosion': {'threshold': 10.0, 'window': 10},
            'low_entropy': {'threshold': 0.5, 'window': 100},
            'low_utilization': {'threshold': 0.1, 'window': 100},
            'high_memory': {'threshold': 0.9, 'window': 10},
        }
        
        self.history = {key: [] for key in self.alerts}
    
    def check_alerts(self, metrics):
        triggered = []
        
        for alert_name, config in self.alerts.items():
            if alert_name in metrics:
                value = metrics[alert_name]
                self.history[alert_name].append(value)
                
                # Check if threshold exceeded
                if len(self.history[alert_name]) >= config['window']:
                    recent = self.history[alert_name][-config['window']:]
                    avg = sum(recent) / len(recent)
                    
                    if alert_name == 'loss_spike':
                        if value > avg * config['threshold']:
                            triggered.append(alert_name)
                    elif alert_name in ['gradient_explosion', 'low_entropy', 'low_utilization']:
                        if avg < config['threshold']:
                            triggered.append(alert_name)
                    elif alert_name == 'high_memory':
                        if value > config['threshold']:
                            triggered.append(alert_name)
        
        return triggered
```

### Alert Actions

```python
def handle_alert(alert_name, metrics, step):
    """Handle triggered alerts."""
    if alert_name == 'loss_spike':
        logger.warning(f"Loss spike detected at step {step}: {metrics['train/loss']:.3f}")
        # Optionally: skip batch, reduce LR, etc.
    
    elif alert_name == 'gradient_explosion':
        logger.warning(f"Gradient explosion at step {step}: {metrics['grad_norm/total']:.3f}")
        # Optionally: clip gradients, skip optimizer step
    
    elif alert_name == 'low_entropy':
        logger.warning(f"Low routing entropy at step {step}: {metrics['router/entropy']:.3f}")
        # Optionally: increase router temperature
    
    elif alert_name == 'low_utilization':
        logger.warning(f"Low expert utilization at step {step}: {metrics['router/utilization']:.3f}")
        # Optionally: increase expert dropout
    
    elif alert_name == 'high_memory':
        logger.warning(f"High memory usage at step {step}: {metrics['gpu/memory_allocated']:.2f} GB")
        # Optionally: reduce batch size, enable checkpointing
```

---

## 7. Configuration

### configs/pretrain.yaml

```yaml
monitoring:
  # Logging backends
  wandb_enabled: true
  wandb_project: fusionllm-pretrain
  wandb_entity: null
  wandb_run_name: null
  
  mlflow_enabled: true
  mlflow_tracking_uri: file:./mlruns
  mlflow_experiment_name: fusionllm-pretrain
  mlflow_run_name: null
  
  # Log intervals
  log_interval: 100
  eval_interval: 1000
  save_interval: 1000
  
  # Metrics to track
  track_gradients: true
  track_activations: true
  track_router: true
  track_performance: true
  track_gpu: true
  
  # Alerting
  alerting_enabled: true
  alert_thresholds:
    loss_spike: 2.0
    gradient_explosion: 10.0
    low_entropy: 0.5
    low_utilization: 0.1
    high_memory: 0.9
```

---

## 8. Checklist

### Phase 1: Core Metrics

- [x] Loss tracking
- [x] Gradient norm tracking
- [x] Learning rate tracking
- [ ] Activation norm tracking

### Phase 2: Router Metrics

- [x] Routing entropy
- [x] Expert utilization
- [ ] Load variance
- [ ] Collapse detection

### Phase 3: Performance

- [x] Throughput tracking
- [ ] GPU utilization
- [ ] Communication overhead

### Phase 4: Dashboard

- [x] W&B integration
- [x] MLflow integration
- [ ] Dashboard layout
- [ ] Alerting system

---

## 9. References

1. W&B: "Weights & Biases" (wandb.ai)
2. MLflow: "MLflow Tracking" (mlflow.org)
3. PyTorch Profiler: "PyTorch Profiler" (PyTorch docs)
4. nvidia-smi: "NVIDIA System Management Interface" (NVIDIA docs)
