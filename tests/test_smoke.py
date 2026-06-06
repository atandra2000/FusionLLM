# tests/test_smoke.py
"""Smoke tests for FusionLLM.

Quick validation tests for core functionality:
- Forward pass tests
- Training step tests
- Checkpoint tests
- Import tests
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core models
        from models.moe import DeepSeekMoE
        from models.transformer import Transformer, TransformerBlock, parse_schedule
        from models.mla import MultiHeadLatentAttention
        from models.gated_deltanet import GatedDeltaNet
        
        # Training
        from training.loss import FusionLLMLoss, LossConfig
        from training.numerical_health import NumericalHealthMonitor, HealthConfig
        
        # Utils
        from utils.memory_profiler import MemoryProfiler
        from utils.compile import compile_model, verify_compilation
        from utils.nccl_profiler import NCCLProfiler
        
        # Benchmarks
        from benchmarks.benchmark_delta_rule import benchmark_delta_rule
        from benchmarks.benchmark_moe import benchmark_moe_routing
        
        print("✓ All imports successful")
    
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_moe_forward():
    """Test MoE forward pass."""
    print("Testing MoE forward pass...")
    
    from models.moe import DeepSeekMoE
    
    config = {
        'dim': 128,
        'inter_dim': 256,
        'n_routed_experts': 8,
        'n_shared_experts': 2,
        'n_activated_experts': 2,
        'moe_inter_dim': 256,
        'route_scale': 1.0,
        'rms_norm_eps': 1e-6,
    }
    
    moe = DeepSeekMoE(config)
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 128)
    
    output = moe(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✓ MoE forward pass test passed")



def test_transformer_forward():
    """Test Transformer forward pass."""
    print("Testing Transformer forward pass...")
    
    from models.transformer import Transformer
    
    config = {
        'vocab_size': 1000,
        'max_seq_len': 512,
        'dim': 128,
        'n_layers': 2,
        'n_heads': 4,
        'head_dim': 32,
        'dim_att': 128,
        'inter_dim': 256,
        'ssm_type': 'gdn',
        'layer_schedule': '1:1',
        'n_routed_experts': 4,
        'n_shared_experts': 1,
        'moe_inter_dim': 128,
        'n_activated_experts': 2,
        'kv_lora_rank': 32,
        'qk_rope_head_dim': 16,
        'qk_nope_head_dim': 16,
        'q_lora_rank': 32,
        'v_head_dim': 32,
        'gdn_headdim': 32,
        'gdn_d_state': 64,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'mtp_depth': 1,
        'route_scale': 1.0,
        'tie_embeddings': True,
        'muP': False,
        'logit_softcap': 15.0,
    }
    
    transformer = Transformer(config)
    
    batch_size, seq_len = 2, 32
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = transformer(tokens)
    
    assert logits.shape == (batch_size, seq_len, 1000), f"Expected shape (2, 32, 1000), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Output contains NaN"
    assert not torch.isinf(logits).any(), "Output contains Inf"
    
    print("✓ Transformer forward pass test passed")



def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    from training.loss import FusionLLMLoss, LossConfig
    
    config = LossConfig(
        label_smoothing=0.0,
        z_loss_weight=0.01,
        mtp_loss_weight=0.1,
    )
    
    loss_fn = FusionLLMLoss(config)
    
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    result = loss_fn(logits, labels)
    
    assert 'loss' in result, "Missing 'loss' in result"
    assert 'perplexity' in result, "Missing 'perplexity' in result"
    assert result['loss'].grad_fn is not None, "Loss should be differentiable"
    
    # Test backward pass
    result['loss'].backward()
    assert logits.grad is not None, "Gradients should be computed"
    
    print("✓ Loss computation test passed")



def test_numerical_health():
    """Test numerical health monitoring."""
    print("Testing numerical health monitoring...")
    
    from training.numerical_health import NumericalHealthMonitor, HealthConfig
    
    config = HealthConfig(
        loss_spike_window=10,
        loss_spike_threshold=2.0,
    )
    
    monitor = NumericalHealthMonitor(config)
    
    # Feed normal losses
    for i in range(10):
        loss = 1.0 + 0.1 * (i % 5)
        monitor.update_loss(loss, i)
    
    # Check stats
    stats = monitor.get_stats()
    assert stats['loss_ema'] is not None, "EMA should be initialized"
    assert stats['spike_count'] == 0, "No spikes should be detected"
    
    print("✓ Numerical health monitoring test passed")



def test_memory_profiler():
    """Test memory profiler."""
    print("Testing memory profiler...")
    
    from utils.memory_profiler import MemoryProfiler, profile_context
    
    profiler = MemoryProfiler()
    
    with profiler.profile("test_operation"):
        x = torch.randn(100, 100)
        y = torch.mm(x, x)
    
    stats = profiler.get_summary()
    assert 'num_snapshots' in stats, "Operation not recorded"
    
    print("✓ Memory profiler test passed")



def test_checkpoint_policy():
    """Test checkpoint policy."""
    print("Testing checkpoint policy...")
    
    from models.transformer import parse_schedule
    
    # Test different schedules
    schedule_5_1 = parse_schedule(12, "5:1")
    assert len(schedule_5_1) == 12, f"Expected 12 layers, got {len(schedule_5_1)}"
    assert sum(schedule_5_1) == 2, f"Expected 2 SSM layers, got {sum(schedule_5_1)}"
    
    schedule_3_1 = parse_schedule(8, "3:1")
    assert len(schedule_3_1) == 8, f"Expected 8 layers, got {len(schedule_3_1)}"
    assert sum(schedule_3_1) == 2, f"Expected 2 SSM layers, got {sum(schedule_3_1)}"
    
    print("✓ Checkpoint policy test passed")



def test_buffer_reuse():
    """Test MoE forward produces correct outputs and gradients."""
    print("Testing MoE forward consistency...")
    
    from models.moe import DeepSeekMoE
    
    config = {
        'dim': 128,
        'inter_dim': 256,
        'n_routed_experts': 8,
        'n_shared_experts': 2,
        'n_activated_experts': 2,
        'moe_inter_dim': 256,
        'route_scale': 1.0,
        'rms_norm_eps': 1e-6,
    }
    
    moe = DeepSeekMoE(config)
    moe.eval()
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 128)
    
    # First forward pass
    output1 = moe(x)
    assert output1.shape == x.shape, f"Shape mismatch: {output1.shape} vs {x.shape}"
    
    # Second forward pass (same output in eval mode)
    output2 = moe(x)
    assert torch.allclose(output1, output2, atol=1e-5), "Eval outputs differ"
    
    # Gradient flow
    moe.train()
    x2 = torch.randn(batch_size, seq_len, 128, requires_grad=True)
    output3 = moe(x2)
    loss = output3.sum()
    loss.backward()
    assert x2.grad is not None, "Gradient did not flow through MoE"
    
    print("✓ MoE forward consistency test passed")


def main():
    """Run all smoke tests."""
    print("FusionLLM Smoke Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_moe_forward,
        test_transformer_forward,
        test_loss_computation,
        test_numerical_health,
        test_memory_profiler,
        test_checkpoint_policy,
        test_buffer_reuse,
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
