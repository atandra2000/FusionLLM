# Training Stability Plan

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: Implementation Guide

---

## Executive Summary

This document outlines the training stability measures for FusionLLM V2, focusing on:
1. **EMA weights** for stable evaluation
2. **DeepNorm** for residual scaling
3. **Router warmup** for MoE stability
4. **Expert collapse detection** for monitoring
5. **Numerical safety** for NaN/Inf prevention

---

## 1. Exponential Moving Average (EMA)

### Implementation

**Location**: `training/ema.py` (to be created)

```python
class EMAManager:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}
    
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param
    
    def apply_shadow(self):
        # Save original params
        self.original = {name: param.clone() for name, param in self.model.named_parameters()}
        # Apply EMA
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self):
        # Restore original params
        for name, param in self.model.named_parameters():
            param.data.copy_(self.original[name])
    
    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
```

### Usage in Training Loop

```python
ema = EMAManager(model, decay=0.9999)

for step in range(total_steps):
    # Training step
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    
    # Update EMA
    ema.update()
    
    # Evaluation (using EMA weights)
    if step % eval_interval == 0:
        ema.apply_shadow()
        eval_metrics = evaluate(model)
        ema.restore()
    
    # Checkpointing
    if step % save_interval == 0:
        save_checkpoint({
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
```

### Configuration

```yaml
# In configs/pretrain.yaml
ema:
  enabled: true
  decay: 0.9999
  save_checkpoint: true
```

---

## 2. DeepNorm Evaluation

### Current Status

**Location**: `models/transformer.py` (to be audited)

DeepNorm applies residual scaling based on network depth:
```
output = α * LayerNorm(x + sublayer(x))
```

where α = (2N)^(1/4) for N layers.

### Implementation Plan

1. **Audit current residual connections** in `TransformerBlock`
2. **Add optional DeepNorm** with configurable α
3. **Benchmark** vs current approach (QK-norm only)

### Code Structure

```python
class TransformerBlock(nn.Module):
    def __init__(self, config, use_deepnorm=False):
        super().__init__()
        # ... existing layers ...
        
        if use_deepnorm:
            N = config['n_layers']
            self.alpha = (2 * N) ** 0.25
        else:
            self.alpha = 1.0
    
    def forward(self, x):
        # Residual with optional DeepNorm scaling
        x = x + self.alpha * self.self_attn(self.norm1(x))
        x = x + self.alpha * self.ffn(self.norm2(x))
        return x
```

### Configuration

```yaml
model:
  use_deepnorm: false  # Enable after benchmarking
  deepnorm_alpha: null  # Auto-computed if null
```

### Testing

- Test with/without DeepNorm on smoke run
- Compare gradient norms
- Check convergence stability

---

## 3. Router Warmup

### Current Status

**Location**: `models/moe.py` (AuxLossFreeGate)

The current implementation includes:
- Bias-based load balancing
- Expert dropout during warmup
- Configurable warmup steps

### Enhancements

#### 3.1 Router Temperature Scheduling

```python
class RouterWarmup:
    def __init__(self, config):
        self.warmup_steps = config.get('moe_warmup_steps', 2000)
        self.initial_temperature = config.get('router_initial_temperature', 2.0)
        self.final_temperature = config.get('router_final_temperature', 1.0)
    
    def get_temperature(self, step):
        if step >= self.warmup_steps:
            return self.final_temperature
        progress = step / self.warmup_steps
        return self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
```

#### 3.2 Shared Expert Emphasis

During early training, increase weight of shared experts:

```python
def compute_shared_expert_weight(self, step):
    if step >= self.warmup_steps:
        return 1.0
    # Linear ramp from 2.0 to 1.0
    return 2.0 - (step / self.warmup_steps)
```

#### 3.3 Load Balancing Warmup

Gradually enable bias updates:

```python
def get_bias_update_speed(self, step):
    if step < self.warmup_steps // 2:
        return 0.0  # No updates
    elif step < self.warmup_steps:
        # Linear ramp
        progress = (step - self.warmup_steps // 2) / (self.warmup_steps // 2)
        return self.bias_update_speed * progress
    else:
        return self.bias_update_speed
```

### Configuration

```yaml
model:
  moe_warmup_steps: 2000
  router_initial_temperature: 2.0
  router_final_temperature: 1.0
  shared_expert_warmup: true
```

---

## 4. Expert Collapse Detection

### Implementation

**Location**: `utils/expert_monitor.py` (to be created)

```python
class ExpertCollapseDetector:
    def __init__(self, config):
        self.n_routed_experts = config['n_routed_experts']
        self.entropy_threshold = config.get('entropy_threshold', 0.5)
        self.utilization_threshold = config.get('utilization_threshold', 0.1)
        
        # Tracking buffers
        self.token_counts = []
        self.entropy_history = []
        self.utilization_history = []
    
    def update(self, routing_stats):
        # Extract metrics
        counts = routing_stats['counts']
        load = routing_stats['load']
        utilisation = routing_stats['utilisation']
        
        # Compute routing entropy
        entropy = -(load * torch.log(load + 1e-10)).sum()
        
        # Store history
        self.token_counts.append(counts)
        self.entropy_history.append(entropy)
        self.utilization_history.append(utilisation)
    
    def check_collapse(self):
        warnings = []
        
        # Check utilization
        recent_util = self.utilization_history[-100:] if len(self.utilization_history) >= 100 else self.utilization_history
        avg_util = sum(recent_util) / len(recent_util)
        
        if avg_util < self.utilization_threshold:
            warnings.append(f"Low expert utilization: {avg_util:.3f}")
        
        # Check entropy
        recent_entropy = self.entropy_history[-100:] if len(self.entropy_history) >= 100 else self.entropy_history
        avg_entropy = sum(recent_entropy) / len(recent_entropy)
        
        if avg_entropy < self.entropy_threshold:
            warnings.append(f"Low routing entropy: {avg_entropy:.3f}")
        
        # Check load variance
        if len(self.token_counts) >= 100:
            recent_counts = torch.stack(self.token_counts[-100:])
            load_variance = recent_counts.var(dim=0).mean()
            if load_variance > 0.5:
                warnings.append(f"High load variance: {load_variance:.3f}")
        
        return warnings
    
    def state_dict(self):
        return {
            'token_counts': self.token_counts,
            'entropy_history': self.entropy_history,
            'utilization_history': self.utilization_history,
        }
```

### Logging

```python
# In training loop
collapse_detector = ExpertCollapseDetector(config)

for step in range(total_steps):
    # ... training step ...
    
    # Update collapse detector
    if step % log_interval == 0:
        stats = moe_layer.get_routing_stats()
        collapse_detector.update(stats)
        
        # Check for collapse
        warnings = collapse_detector.check_collapse()
        for w in warnings:
            logger.warning(f"Expert collapse warning: {w}")
        
        # Log metrics
        logger.log({
            'expert/utilization': stats['utilisation'],
            'expert/entropy': collapse_detector.entropy_history[-1],
            'expert/load_variance': stats['counts'].float().var(),
        })
```

### Configuration

```yaml
expert_monitor:
  enabled: true
  entropy_threshold: 0.5
  utilization_threshold: 0.1
  check_interval: 100
```

---

## 5. Numerical Safety Layer

### Implementation

**Location**: `utils/numerical_safety.py` (to be created)

```python
class NumericalSafetyLayer:
    def __init__(self, config):
        self.nan_check_enabled = config.get('nan_check', True)
        self.inf_check_enabled = config.get('inf_check', True)
        self.grad_explosion_threshold = config.get('grad_explosion_threshold', 10.0)
        self.activation_explosion_threshold = config.get('activation_explosion_threshold', 100.0)
        
        # Tracking
        self.nan_count = 0
        self.inf_count = 0
        self.grad_explosion_count = 0
        self.activation_explosion_count = 0
    
    def check_tensor(self, tensor, name, step):
        if self.nan_check_enabled and torch.isnan(tensor).any():
            self.nan_count += 1
            raise RuntimeError(f"NaN detected in {name} at step {step}")
        
        if self.inf_check_enabled and torch.isinf(tensor).any():
            self.inf_count += 1
            raise RuntimeError(f"Inf detected in {name} at step {step}")
    
    def check_gradients(self, model, step):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > self.grad_explosion_threshold:
                    self.grad_explosion_count += 1
                    warnings.warn(f"Gradient explosion in {name}: {grad_norm:.2f}")
    
    def check_activations(self, activations, step):
        for name, act in activations.items():
            act_norm = act.norm().item()
            if act_norm > self.activation_explosion_threshold:
                self.activation_explosion_count += 1
                warnings.warn(f"Activation explosion in {name}: {act_norm:.2f}")
    
    def safety_check(self, loss, model, activations, step):
        # Check loss
        self.check_tensor(loss, 'loss', step)
        
        # Check gradients
        self.check_gradients(model, step)
        
        # Check activations
        self.check_activations(activations, step)
        
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'grad_explosion_count': self.grad_explosion_count,
            'activation_explosion_count': self.activation_explosion_count,
        }
```

### Usage in Training Loop

```python
safety_layer = NumericalSafetyLayer(config)

for step in range(total_steps):
    # Forward pass
    with autocast():
        output = model(batch)
        activations = output.activations  # Capture activations
    
    loss = compute_loss(output, batch)
    
    # Safety check before backward
    safety_stats = safety_layer.safety_check(loss, model, activations, step)
    
    # Backward pass
    loss.backward()
    
    # Gradient check
    safety_layer.check_gradients(model, step)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Log safety metrics
    if step % log_interval == 0:
        logger.log(safety_stats)
```

### Configuration

```yaml
numerical_safety:
  nan_check: true
  inf_check: true
  grad_explosion_threshold: 10.0
  activation_explosion_threshold: 100.0
  fail_loud: true  # Raise exception on failure
```

---

## 6. Integration Checklist

### Phase 1: Core Stability

- [ ] Implement EMA manager
- [ ] Add DeepNorm option
- [ ] Enhance router warmup
- [ ] Create expert collapse detector
- [ ] Add numerical safety layer

### Phase 2: Testing

- [ ] Test EMA checkpoint save/load
- [ ] Benchmark DeepNorm vs current approach
- [ ] Validate router warmup on smoke run
- [ ] Test collapse detection on edge cases
- [ ] Verify numerical safety catches NaN/Inf

### Phase 3: Production

- [ ] Integrate all components into training loop
- [ ] Add logging for all stability metrics
- [ ] Create monitoring dashboard
- [ ] Document failure modes and recovery

---

## 7. Monitoring Dashboard

### Key Metrics to Track

| Metric | Threshold | Action |
|--------|-----------|--------|
| Expert utilization | <0.1 | Warning |
| Routing entropy | <0.5 | Warning |
| Gradient norm | >10.0 | Warning |
| Loss NaN | Any | Fail |
| Loss Inf | Any | Fail |
| EMA decay | N/A | Log |

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│ Training Stability Dashboard                           │
├─────────────────────────────────────────────────────────┤
│ Loss: 2.345  │  Grad Norm: 0.823  │  LR: 3e-4         │
├─────────────────────────────────────────────────────────┤
│ Expert Utilization: 0.45  │  Routing Entropy: 1.23     │
├─────────────────────────────────────────────────────────┤
│ NaN Count: 0  │  Inf Count: 0  │  Grad Explosion: 0    │
├─────────────────────────────────────────────────────────┤
│ EMA Decay: 0.9999  │  EMA Loss: 2.340                 │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Failure Recovery

### NaN/Inf Detection

If NaN/Inf detected:
1. Save checkpoint before failure
2. Log detailed diagnostics
3. Optionally: skip batch and continue
4. Or: fail loudly and alert

### Expert Collapse Recovery

If collapse detected:
1. Increase router temperature
2. Increase expert dropout
3. Reset bias terms
4. Log warning

### Gradient Explosion Recovery

If gradient norm > threshold:
1. Log gradient norms per layer
2. Optionally: skip optimizer step
3. Reduce learning rate
4. Alert and continue

---

## 9. References

1. EMA: "Exponential Moving Average for Model Training" (PyTorch docs)
2. DeepNorm: "DeepNet: Scaling Transformers to 1,000 Layers" (2022)
3. Router Warmup: "Switch Transformers" (2022)
4. Expert Collapse: "Mixture of Experts with Expert Choice Routing" (2022)
5. Numerical Safety: "Gradient Clipping" (Goodfellow et al.)
