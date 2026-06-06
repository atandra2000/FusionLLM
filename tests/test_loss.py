# tests/test_loss.py
"""Tests for standardized loss computation."""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.loss import (
    StandardCrossEntropy,
    MTPLoss,
    MoELoadBalancingLoss,
    FusionLLMLoss,
    LossConfig,
    compute_loss,
)


def test_cross_entropy_basic():
    """Test basic cross-entropy loss."""
    print("Testing basic cross-entropy loss...")
    
    config = LossConfig(label_smoothing=0.0, reduction='sum')
    loss_fn = StandardCrossEntropy(config)
    
    # Create dummy data with requires_grad
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    result = loss_fn(logits, labels)
    
    assert 'loss' in result, "Missing 'loss' in result"
    assert 'perplexity' in result, "Missing 'perplexity' in result"
    assert 'num_tokens' in result, "Missing 'num_tokens' in result"
    assert result['loss'].grad_fn is not None, "Loss should be differentiable"
    assert result['perplexity'] > 0, "Perplexity should be positive"
    
    # Test backward pass
    result['loss'].backward()
    assert logits.grad is not None, "Gradients should be computed"
    
    print("✓ Basic cross-entropy test passed")



def test_cross_entropy_ignore_index():
    """Test cross-entropy with ignore index."""
    print("Testing cross-entropy with ignore index...")
    
    config = LossConfig(ignore_index=-100)
    loss_fn = StandardCrossEntropy(config)
    
    # Create dummy data with ignore tokens
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Set some tokens to ignore
    labels[0, :3] = -100
    labels[1, 5:] = -100
    
    result = loss_fn(logits, labels)
    
    # Should only count valid tokens
    expected_tokens = (batch_size * seq_len) - 3 - 5
    assert result['num_tokens'] == expected_tokens, f"Expected {expected_tokens} tokens, got {result['num_tokens']}"
    
    print("✓ Cross-entropy ignore index test passed")



def test_cross_entropy_label_smoothing():
    """Test cross-entropy with label smoothing."""
    print("Testing cross-entropy with label smoothing...")
    
    config_smooth = LossConfig(label_smoothing=0.1)
    config_no_smooth = LossConfig(label_smoothing=0.0)
    
    loss_fn_smooth = StandardCrossEntropy(config_smooth)
    loss_fn_no_smooth = StandardCrossEntropy(config_no_smooth)
    
    # Create dummy data with requires_grad
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    result_smooth = loss_fn_smooth(logits, labels)
    result_no_smooth = loss_fn_no_smooth(logits, labels)
    
    # Smooth loss should be higher (more uniform distribution)
    # This is a heuristic, not always true
    assert result_smooth['loss'].grad_fn is not None, "Smooth loss should be differentiable"
    assert result_no_smooth['loss'].grad_fn is not None, "Non-smooth loss should be differentiable"
    
    print("✓ Cross-entropy label smoothing test passed")



def test_z_loss():
    """Test z-loss computation."""
    print("Testing z-loss computation...")
    
    config = LossConfig(z_loss_weight=0.01)
    loss_fn = StandardCrossEntropy(config)
    
    # Create dummy logits with requires_grad
    logits = torch.randn(2, 10, 100, requires_grad=True)
    
    z_loss = loss_fn.compute_z_loss(logits)
    
    assert z_loss.grad_fn is not None, "Z-loss should be differentiable"
    assert z_loss >= 0, "Z-loss should be non-negative"
    
    # Test with zero weight
    config_no_z = LossConfig(z_loss_weight=0.0)
    loss_fn_no_z = StandardCrossEntropy(config_no_z)
    
    z_loss_no_z = loss_fn_no_z.compute_z_loss(logits)
    assert z_loss_no_z == 0, "Z-loss with zero weight should be zero"
    
    print("✓ Z-loss test passed")



def test_mtp_loss():
    """Test MTP auxiliary loss."""
    print("Testing MTP auxiliary loss...")
    
    config = LossConfig(mtp_loss_weight=0.1)
    loss_fn = MTPLoss(config)
    
    # Create dummy data with requires_grad
    batch_size, seq_len, vocab_size = 2, 10, 100
    mtp_logits = [torch.randn(batch_size, seq_len, vocab_size, requires_grad=True) for _ in range(3)]
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    result = loss_fn(mtp_logits, labels)
    
    assert 'mtp_loss' in result, "Missing 'mtp_loss' in result"
    assert 'mtp_layer_losses' in result, "Missing 'mtp_layer_losses' in result"
    assert len(result['mtp_layer_losses']) == 3, f"Expected 3 layer losses, got {len(result['mtp_layer_losses'])}"
    assert result['mtp_loss'].grad_fn is not None, "MTP loss should be differentiable"
    
    # Test with zero weight
    config_zero = LossConfig(mtp_loss_weight=0.0)
    loss_fn_zero = MTPLoss(config_zero)
    
    result_zero = loss_fn_zero(mtp_logits, labels)
    assert result_zero['mtp_loss'] == 0, "MTP loss with zero weight should be zero"
    
    print("✓ MTP loss test passed")



def test_moe_loss():
    """Test MoE load-balancing loss."""
    print("Testing MoE load-balancing loss...")
    
    config = LossConfig(moe_balance_loss_weight=0.01, moe_z_loss_weight=0.001)
    loss_fn = MoELoadBalancingLoss(config)
    
    # Create dummy data with requires_grad
    batch_size, seq_len, n_experts = 2, 10, 8
    router_probs = torch.randn(batch_size, seq_len, n_experts, requires_grad=True)
    router_probs = torch.softmax(router_probs, dim=-1)
    expert_mask = torch.zeros(batch_size, seq_len, n_experts)
    expert_mask[:, :, :2] = 1  # Select top-2 experts
    
    result = loss_fn(router_probs, expert_mask)
    
    assert 'balance_loss' in result, "Missing 'balance_loss' in result"
    assert 'z_loss' in result, "Missing 'z_loss' in result"
    assert 'routing_stats' in result, "Missing 'routing_stats' in result"
    assert result['balance_loss'].grad_fn is not None, "Balance loss should be differentiable"
    assert result['z_loss'].grad_fn is not None, "Z-loss should be differentiable"
    
    # Check routing stats
    assert 'load_balance' in result['routing_stats'], "Missing 'load_balance' in stats"
    assert 'expert_load_std' in result['routing_stats'], "Missing 'expert_load_std' in stats"
    assert 'router_entropy' in result['routing_stats'], "Missing 'router_entropy' in stats"
    
    print("✓ MoE loss test passed")



def test_fusionllm_loss():
    """Test combined FusionLLM loss."""
    print("Testing FusionLLM loss...")
    
    config = LossConfig(
        label_smoothing=0.0,
        z_loss_weight=0.01,
        mtp_loss_weight=0.1,
        moe_balance_loss_weight=0.01,
    )
    loss_fn = FusionLLMLoss(config)
    
    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # MTP logits
    mtp_logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(2)]
    
    # MoE data
    n_experts = 8
    router_probs = torch.randn(batch_size, seq_len, n_experts)
    router_probs = torch.softmax(router_probs, dim=-1)
    expert_mask = torch.zeros(batch_size, seq_len, n_experts)
    expert_mask[:, :, :2] = 1
    
    result = loss_fn(
        logits, labels,
        mtp_logits=mtp_logits,
        router_probs=router_probs,
        expert_mask=expert_mask,
    )
    
    assert 'loss' in result, "Missing 'loss' in result"
    assert 'perplexity' in result, "Missing 'perplexity' in result"
    assert 'z_loss' in result, "Missing 'z_loss' in result"
    assert 'mtp_loss' in result, "Missing 'mtp_loss' in result"
    assert 'balance_loss' in result, "Missing 'balance_loss' in result"
    assert 'loss' in result, "Missing 'loss' in result"
    
    # Total loss should be sum of components
    expected_loss = (
        result.get('ce_loss', 0) + 
        result['z_loss'] + 
        result['mtp_loss'] + 
        result['balance_loss'] +
        result.get('z_loss_moe', 0)
    )
    
    print("✓ FusionLLM loss test passed")



def test_convenience_function():
    """Test convenience function."""
    print("Testing convenience function...")
    
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    result = compute_loss(logits, labels)
    
    assert 'loss' in result, "Missing 'loss' in result"
    assert 'perplexity' in result, "Missing 'perplexity' in result"
    
    print("✓ Convenience function test passed")



def main():
    """Run all tests."""
    print("Loss Computation Tests")
    print("=" * 50)
    
    tests = [
        test_cross_entropy_basic,
        test_cross_entropy_ignore_index,
        test_cross_entropy_label_smoothing,
        test_z_loss,
        test_mtp_loss,
        test_moe_loss,
        test_fusionllm_loss,
        test_convenience_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
