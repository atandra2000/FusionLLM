# tests/test_convergence.py
"""Convergence validation suite for FusionLLM.

Provides tools to validate training convergence:
- Loss curve analysis
- Gradient flow verification
- Learning rate schedule validation
- Performance regression detection
"""

import torch
import torch.nn as nn
import math
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence analysis."""
    loss_values: List[float]
    gradient_norms: List[float]
    learning_rates: List[float]
    step_numbers: List[int]


class ConvergenceValidator:
    """Validates training convergence."""
    
    def __init__(self):
        self.metrics = ConvergenceMetrics(
            loss_values=[],
            gradient_norms=[],
            learning_rates=[],
            step_numbers=[],
        )
    
    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        learning_rate: float,
    ):
        """Update metrics."""
        self.metrics.step_numbers.append(step)
        self.metrics.loss_values.append(loss)
        self.metrics.gradient_norms.append(grad_norm)
        self.metrics.learning_rates.append(learning_rate)
    
    def check_loss_convergence(
        self,
        min_steps: int = 100,
        window_size: int = 50,
        max_increase_ratio: float = 0.1,
    ) -> Dict:
        """Check if loss is converging.
        
        Args:
            min_steps: Minimum steps before checking
            window_size: Window size for moving average
            max_increase_ratio: Maximum allowed increase ratio
            
        Returns:
            Dictionary with convergence check results
        """
        if len(self.metrics.loss_values) < min_steps:
            return {"converged": None, "reason": "Insufficient data"}
        
        losses = self.metrics.loss_values
        
        # Calculate moving averages
        recent_avg = sum(losses[-window_size:]) / window_size
        older_avg = sum(losses[-2*window_size:-window_size]) / window_size
        
        # Check for divergence
        if older_avg > 0:
            increase_ratio = (recent_avg - older_avg) / older_avg
        else:
            increase_ratio = float('inf')
        
        is_converging = increase_ratio < max_increase_ratio
        
        return {
            "converged": is_converging,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "increase_ratio": increase_ratio,
            "reason": "Loss is decreasing" if is_converging else "Loss may be diverging",
        }
    
    def check_gradient_flow(self) -> Dict:
        """Check gradient flow health.
        
        Returns:
            Dictionary with gradient flow analysis
        """
        if not self.metrics.gradient_norms:
            return {"healthy": None, "reason": "No gradient data"}
        
        norms = self.metrics.gradient_norms
        
        # Check for vanishing gradients
        avg_norm = sum(norms[-100:]) / min(100, len(norms))
        min_norm = min(norms[-100:]) if len(norms) >= 100 else min(norms)
        
        # Check for exploding gradients
        max_norm = max(norms[-100:]) if len(norms) >= 100 else max(norms)
        
        # Heuristics
        has_vanishing = avg_norm < 1e-6
        has_exploding = max_norm > 100.0
        is_healthy = not has_vanishing and not has_exploding
        
        return {
            "healthy": is_healthy,
            "avg_norm": avg_norm,
            "min_norm": min_norm,
            "max_norm": max_norm,
            "has_vanishing": has_vanishing,
            "has_exploding": has_exploding,
            "reason": "Gradient flow healthy" if is_healthy else (
                "Vanishing gradients" if has_vanishing else "Exploding gradients"
            ),
        }
    
    def check_learning_rate_schedule(
        self,
        expected_warmup_steps: Optional[int] = None,
        expected_total_steps: Optional[int] = None,
    ) -> Dict:
        """Validate learning rate schedule.
        
        Args:
            expected_warmup_steps: Expected warmup steps
            expected_total_steps: Expected total steps
            
        Returns:
            Dictionary with schedule validation
        """
        if not self.metrics.learning_rates:
            return {"valid": None, "reason": "No learning rate data"}
        
        lrs = self.metrics.learning_rates
        
        # Check for monotonic decrease after warmup
        if len(lrs) > 10:
            recent_lrs = lrs[-10:]
            is_decreasing = all(recent_lrs[i] >= recent_lrs[i+1] for i in range(len(recent_lrs)-1))
        else:
            is_decreasing = True
        
        # Check for zero learning rate
        has_zero_lr = any(lr == 0 for lr in lrs[-100:]) if len(lrs) >= 100 else False
        
        return {
            "valid": is_decreasing and not has_zero_lr,
            "current_lr": lrs[-1] if lrs else 0,
            "min_lr": min(lrs) if lrs else 0,
            "max_lr": max(lrs) if lrs else 0,
            "is_decreasing": is_decreasing,
            "has_zero_lr": has_zero_lr,
            "reason": "Schedule looks valid" if is_decreasing and not has_zero_lr else "Invalid schedule",
        }
    
    def generate_report(self) -> str:
        """Generate convergence report."""
        report = ["=" * 60]
        report.append("Convergence Validation Report")
        report.append("=" * 60)
        
        # Loss convergence
        loss_check = self.check_loss_convergence()
        report.append(f"\nLoss Convergence:")
        report.append(f"  Status: {'✓ Converging' if loss_check['converged'] else '✗ May be diverging' if loss_check['converged'] is False else '? Insufficient data'}")
        if loss_check.get('recent_avg'):
            report.append(f"  Recent avg: {loss_check['recent_avg']:.4f}")
            report.append(f"  Older avg: {loss_check['older_avg']:.4f}")
            report.append(f"  Change: {loss_check['increase_ratio']:.4f}")
        
        # Gradient flow
        grad_check = self.check_gradient_flow()
        report.append(f"\nGradient Flow:")
        report.append(f"  Status: {'✓ Healthy' if grad_check['healthy'] else '✗ ' + grad_check['reason']}")
        if grad_check.get('avg_norm'):
            report.append(f"  Avg norm: {grad_check['avg_norm']:.4f}")
            report.append(f"  Min norm: {grad_check['min_norm']:.4f}")
            report.append(f"  Max norm: {grad_check['max_norm']:.4f}")
        
        # Learning rate schedule
        lr_check = self.check_learning_rate_schedule()
        report.append(f"\nLearning Rate Schedule:")
        report.append(f"  Status: {'✓ Valid' if lr_check['valid'] else '✗ Invalid'}")
        if lr_check.get('current_lr'):
            report.append(f"  Current LR: {lr_check['current_lr']:.6f}")
            report.append(f"  Min LR: {lr_check['min_lr']:.6f}")
            report.append(f"  Max LR: {lr_check['max_lr']:.6f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def test_convergence_validator():
    """Test convergence validator."""
    print("Testing convergence validator...")
    
    validator = ConvergenceValidator()
    
    # Simulate training
    for step in range(200):
        loss = 2.0 / (step + 1) + 0.1 * torch.randn(1).item()
        grad_norm = 1.0 + 0.1 * torch.randn(1).item()
        lr = 3e-4 * (1 - step / 200)
        
        validator.update(step, loss, grad_norm, lr)
    
    # Check convergence
    loss_check = validator.check_loss_convergence()
    assert loss_check['converged'] is not None, "Should have enough data"
    
    grad_check = validator.check_gradient_flow()
    assert grad_check['healthy'] is not None, "Should have gradient data"
    
    lr_check = validator.check_learning_rate_schedule()
    assert lr_check['valid'] is not None, "Should have LR data"
    
    # Generate report
    report = validator.generate_report()
    assert len(report) > 0, "Report should not be empty"
    
    print("✓ Convergence validator test passed")
    print(f"\nSample Report:\n{report}")



def main():
    """Run convergence validation tests."""
    print("Convergence Validation Suite")
    print("=" * 50)
    
    tests = [
        test_convergence_validator,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
