# tests/test_e2e_training.py
"""End-to-end training validation tests.

Validates complete training pipeline:
- Forward + backward pass
- Loss computation
- Gradient flow
- Optimizer step
- Checkpoint save/load
"""

import torch
import torch.nn as nn
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import Transformer
from training.loss import FusionLLMLoss, LossConfig
from training.numerical_health import NumericalHealthMonitor, HealthConfig
from training.normuon import NorMuon


def create_training_config():
    """Create minimal training configuration."""
    return {
        'vocab_size': 1000,
        'max_seq_len': 512,
        'dim': 64,
        'n_layers': 2,
        'n_heads': 4,
        'head_dim': 16,
        'dim_att': 64,
        'inter_dim': 128,
        'ssm_type': 'gdn',
        'layer_schedule': '1:1',
        'n_routed_experts': 4,
        'n_shared_experts': 1,
        'moe_inter_dim': 64,
        'n_activated_experts': 2,
        'kv_lora_rank': 16,
        'qk_rope_head_dim': 8,
        'qk_nope_head_dim': 8,
        'q_lora_rank': 16,
        'v_head_dim': 16,
        'gdn_headdim': 16,
        'gdn_d_state': 32,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'mtp_depth': 0,
        'route_scale': 1.0,
        'tie_embeddings': True,
        'muP': False,
        'logit_softcap': 15.0,
        'checkpoint_mla_ratio': 0.0,  # Disable MLA checkpointing for tests
    }


def create_model(config, use_checkpoint=False):
    """Create model with optional checkpointing disabled."""
    return Transformer(config, use_checkpoint=use_checkpoint)


def test_single_training_step():
    """Test single training step."""
    print("Testing single training step...")
    
    config = create_training_config()
    model = create_model(config)
    
    # Create loss and optimizer
    loss_fn = FusionLLMLoss(LossConfig())
    optimizer = NorMuon(model.parameters(), lr=1e-3)
    
    # Create dummy data
    batch_size, seq_len = 2, 32
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    model.train()
    logits = model(tokens)
    
    # Compute loss
    loss_result = loss_fn(logits, targets)
    loss = loss_result['loss']
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed"
    
    # Check gradients are finite
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    
    assert all(g < float('inf') for g in grad_norms), "Inf gradients detected"
    assert all(g == g for g in grad_norms), "NaN gradients detected"
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    print("✓ Single training step test passed")


def test_multi_step_training():
    """Test multi-step training loop."""
    print("Testing multi-step training...")
    
    config = create_training_config()
    model = create_model(config)
    
    loss_fn = FusionLLMLoss(LossConfig())
    optimizer = NorMuon(model.parameters(), lr=1e-3)
    
    batch_size, seq_len = 2, 32
    n_steps = 10
    
    losses = []
    
    for step in range(n_steps):
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        optimizer.zero_grad()
        
        logits = model(tokens)
        loss_result = loss_fn(logits, targets)
        loss = loss_result['loss']
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Check loss is finite
    assert all(l == l for l in losses), "NaN loss detected"
    assert all(l < float('inf') for l in losses), "Inf loss detected"
    
    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    
    print(f"✓ Multi-step training test passed (loss: {avg_first:.4f} -> {avg_last:.4f})")
    return


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    print("Testing checkpoint save/load...")
    
    config = create_training_config()
    model = create_model(config)
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, ckpt_path)
        
        # Load into new model
        model2 = create_model(config)
        checkpoint = torch.load(ckpt_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify models are identical
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"
    
    print("✓ Checkpoint save/load test passed")
    return


def test_health_monitor_integration():
    """Test health monitor integration."""
    print("Testing health monitor integration...")
    
    config = create_training_config()
    model = create_model(config)
    
    # Create health monitor
    health_config = HealthConfig(
        loss_spike_window=5,
        loss_spike_threshold=2.0,
    )
    monitor = NumericalHealthMonitor(health_config)
    
    loss_fn = FusionLLMLoss(LossConfig())
    optimizer = NorMuon(model.parameters(), lr=1e-3)
    
    batch_size, seq_len = 2, 32
    
    # Simulate training with health monitoring
    for step in range(10):
        optimizer.zero_grad()  # Zero gradients before forward pass
        
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        logits = model(tokens)
        loss_result = loss_fn(logits, targets)
        loss = loss_result['loss']
        
        # Update health monitor
        monitor.update_loss(loss.item(), step)
        
        loss.backward()
        
        # Update gradient monitor
        monitor.update_gradients(model, step)
        
        optimizer.step()
    
    # Check stats
    stats = monitor.get_stats()
    assert 'loss_ema' in stats, "Loss EMA not tracked"
    assert 'spike_count' in stats, "Spike count not tracked"
    
    print("✓ Health monitor integration test passed")
    return


def test_numerical_stability():
    """Test numerical stability."""
    print("Testing numerical stability...")
    
    config = create_training_config()
    model = create_model(config)
    
    loss_fn = FusionLLMLoss(LossConfig())
    optimizer = NorMuon(model.parameters(), lr=1e-3)
    
    batch_size, seq_len = 2, 32
    
    # Run training and check for numerical issues
    for step in range(20):
        optimizer.zero_grad()  # Zero gradients before forward pass
        
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        logits = model(tokens)
        loss_result = loss_fn(logits, targets)
        loss = loss_result['loss']
        
        # Check loss is finite
        assert torch.isfinite(loss), f"Loss is not finite at step {step}: {loss.item()}"
        
        loss.backward()
        
        # Check gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        
        optimizer.step()
    
    print("✓ Numerical stability test passed")
    return


def main():
    """Run end-to-end training tests."""
    print("End-to-End Training Validation")
    print("=" * 50)
    
    tests = [
        test_single_training_step,
        test_multi_step_training,
        test_checkpoint_save_load,
        test_health_monitor_integration,
        test_numerical_stability,
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
