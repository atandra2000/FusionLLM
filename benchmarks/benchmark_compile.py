# benchmarks/benchmark_compile.py
"""Benchmark torch.compile performance for FusionLLM.

Profiles compiled vs uncompiled inference and training:
- Forward pass latency
- Memory usage
- Throughput (tokens/sec)
- MFU estimation
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import Transformer, parse_schedule


def create_test_config(
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    vocab_size: int = 1000,
    max_seq_len: int = 512,
) -> dict:
    """Create test configuration for benchmarking."""
    return {
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'head_dim': dim // n_heads,
        'dim_att': dim,
        'inter_dim': dim * 4,
        'ssm_type': 'gdn',
        'layer_schedule': '1:1',
        'n_routed_experts': 4,
        'n_shared_experts': 1,
        'moe_inter_dim': dim,
        'n_activated_experts': 2,
        'kv_lora_rank': dim // 4,
        'qk_rope_head_dim': dim // (n_heads * 2),
        'qk_nope_head_dim': dim // (n_heads * 2),
        'q_lora_rank': dim // 4,
        'v_head_dim': dim // n_heads,
        'gdn_headdim': dim // n_heads,
        'gdn_d_state': 64,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'mtp_depth': 0,
        'route_scale': 1.0,
        'tie_embeddings': True,
        'muP': False,
        'logit_softcap': 15.0,
    }


def benchmark_forward(
    model: nn.Module,
    tokens: torch.Tensor,
    n_warmup: int = 3,
    n_bench: int = 10,
    device: str = 'cpu',
) -> dict:
    """Benchmark forward pass.
    
    Args:
        model: Model to benchmark
        tokens: Input tokens
        n_warmup: Warmup iterations
        n_bench: Benchmark iterations
        device: Device
        
    Returns:
        Dictionary with timing results
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(tokens)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_bench):
            start = time.perf_counter()
            _ = model(tokens)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Throughput
    tokens_per_sec = tokens.numel() / (avg_time / 1000)
    
    return {
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'tokens_per_sec': tokens_per_sec,
        'n_tokens': tokens.numel(),
    }


def benchmark_compile_performance(
    dim: int = 128,
    n_layers: int = 4,
    seq_lens: list = [32, 64, 128],
    device: str = 'cpu',
) -> dict:
    """Benchmark torch.compile performance.
    
    Args:
        dim: Model dimension
        n_layers: Number of layers
        seq_lens: Sequence lengths to test
        device: Device
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking torch.compile (dim={dim}, layers={n_layers})")
    print("=" * 60)
    
    config = create_test_config(dim=dim, n_layers=n_layers)
    
    # Create model
    model = Transformer(config).to(device)
    model.eval()
    
    results = {}
    
    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")
        print("-" * 40)
        
        tokens = torch.randint(0, 1000, (1, seq_len), device=device)
        
        # Uncompiled
        print("  Uncompiled: ", end="", flush=True)
        uncompiled_result = benchmark_forward(model, tokens, device=device)
        print(f"{uncompiled_result['avg_ms']:.2f} ms")
        
        # Compiled
        if hasattr(torch, 'compile'):
            print("  Compiled:   ", end="", flush=True)
            try:
                compiled_model = torch.compile(model, mode='max-autotune', dynamic=True)
                compiled_result = benchmark_forward(compiled_model, tokens, device=device)
                print(f"{compiled_result['avg_ms']:.2f} ms")
                
                speedup = uncompiled_result['avg_ms'] / compiled_result['avg_ms']
                print(f"  Speedup:    {speedup:.2f}x")
                
                results[seq_len] = {
                    'uncompiled': uncompiled_result,
                    'compiled': compiled_result,
                    'speedup': speedup,
                }
            except Exception as e:
                print(f"Failed: {e}")
                results[seq_len] = {
                    'uncompiled': uncompiled_result,
                    'compiled': None,
                    'speedup': None,
                }
        else:
            print("  torch.compile not available")
            results[seq_len] = {
                'uncompiled': uncompiled_result,
                'compiled': None,
                'speedup': None,
            }
    
    return results


def main():
    """Run compile benchmarks."""
    print("FusionLLM torch.compile Benchmark")
    print("=" * 60)
    
    # Test on CPU (GPU available in production)
    device = 'cpu'
    
    # Small model for quick testing
    results = benchmark_compile_performance(
        dim=128,
        n_layers=4,
        seq_lens=[32, 64, 128],
        device=device,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for seq_len, result in results.items():
        print(f"\nSeqLen {seq_len}:")
        print(f"  Uncompiled: {result['uncompiled']['avg_ms']:.2f} ms")
        if result['compiled']:
            print(f"  Compiled:   {result['compiled']['avg_ms']:.2f} ms")
            print(f"  Speedup:    {result['speedup']:.2f}x")
        print(f"  Throughput: {result['uncompiled']['tokens_per_sec']:.0f} tokens/sec")
    
    return results


if __name__ == "__main__":
    main()
