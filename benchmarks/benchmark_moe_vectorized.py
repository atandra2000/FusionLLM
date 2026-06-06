# benchmarks/benchmark_moe_vectorized.py
"""Benchmark MoE vectorized scatter-gather operations.

Compares original vs vectorized implementations:
- Routing overhead
- Expert computation
- Scatter-gather operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe import DeepSeekMoE


def create_moe_config(
    dim: int = 128,
    n_routed_experts: int = 8,
    n_activated_experts: int = 2,
) -> dict:
    """Create MoE configuration for benchmarking."""
    return {
        'dim': dim,
        'inter_dim': dim * 4,
        'n_routed_experts': n_routed_experts,
        'n_shared_experts': 2,
        'n_activated_experts': n_activated_experts,
        'moe_inter_dim': dim * 2,
        'route_scale': 1.0,
        'rms_norm_eps': 1e-6,
    }


def benchmark_moe_forward(
    moe: DeepSeekMoE,
    x: torch.Tensor,
    n_warmup: int = 3,
    n_bench: int = 10,
) -> dict:
    """Benchmark MoE forward pass.
    
    Args:
        moe: MoE module
        x: Input tensor
        n_warmup: Warmup iterations
        n_bench: Benchmark iterations
        
    Returns:
        Dictionary with timing results
    """
    moe.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = moe(x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_bench):
            start = time.perf_counter()
            _ = moe(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    
    return {
        'avg_ms': avg_time,
        'min_ms': min(times),
        'max_ms': max(times),
    }


def benchmark_moe_scaling(
    dim: int = 128,
    batch_sizes: list = [1, 2, 4],
    seq_lens: list = [32, 64, 128],
) -> dict:
    """Benchmark MoE scaling with different input sizes.
    
    Args:
        dim: Model dimension
        batch_sizes: Batch sizes to test
        seq_lens: Sequence lengths to test
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking MoE Scaling (dim={dim})")
    print("=" * 60)
    
    config = create_moe_config(dim=dim)
    moe = DeepSeekMoE(config)
    moe.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\nBatch={batch_size}, SeqLen={seq_len}")
            print("-" * 40)
            
            x = torch.randn(batch_size, seq_len, dim)
            
            result = benchmark_moe_forward(moe, x)
            result['batch_size'] = batch_size
            result['seq_len'] = seq_len
            result['n_tokens'] = batch_size * seq_len
            
            print(f"  Time:     {result['avg_ms']:.2f} ms")
            print(f"  Per token: {result['avg_ms'] / result['n_tokens'] * 1000:.2f} us")
            
            results[(batch_size, seq_len)] = result
    
    return results


def benchmark_moe_vs_dense(
    dim: int = 128,
    n_layers: int = 4,
    seq_len: int = 64,
) -> dict:
    """Benchmark MoE vs dense FFN.
    
    Args:
        dim: Model dimension
        n_layers: Number of layers
        seq_len: Sequence length
        
    Returns:
        Dictionary with comparison results
    """
    print(f"\nBenchmarking MoE vs Dense FFN (dim={dim}, layers={n_layers})")
    print("=" * 60)
    
    # MoE
    moe_config = create_moe_config(dim=dim)
    moe = DeepSeekMoE(moe_config)
    moe.eval()
    
    # Dense FFN
    dense_ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.SiLU(),
        nn.Linear(dim * 4, dim),
    )
    
    x = torch.randn(1, seq_len, dim)
    
    # Benchmark MoE
    moe_result = benchmark_moe_forward(moe, x)
    
    # Benchmark Dense
    dense_times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            _ = dense_ffn(x)
            end = time.perf_counter()
            dense_times.append((end - start) * 1000)
    
    dense_avg = sum(dense_times) / len(dense_times)
    
    print(f"\nResults:")
    print(f"  MoE:   {moe_result['avg_ms']:.2f} ms")
    print(f"  Dense: {dense_avg:.2f} ms")
    print(f"  Ratio: {moe_result['avg_ms'] / dense_avg:.2f}x")
    
    return {
        'moe': moe_result,
        'dense': {'avg_ms': dense_avg},
        'ratio': moe_result['avg_ms'] / dense_avg,
    }


def main():
    """Run MoE benchmarks."""
    print("FusionLLM MoE Vectorized Benchmark")
    print("=" * 60)
    
    # Scaling benchmark
    scaling_results = benchmark_moe_scaling(
        dim=128,
        batch_sizes=[1, 2],
        seq_lens=[32, 64],
    )
    
    # MoE vs Dense comparison
    comparison_results = benchmark_moe_vs_dense(
        dim=128,
        n_layers=4,
        seq_len=64,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\nScaling Results:")
    for (batch_size, seq_len), result in scaling_results.items():
        print(f"  Batch={batch_size}, SeqLen={seq_len}: {result['avg_ms']:.2f} ms")
    
    print(f"\nMoE vs Dense:")
    print(f"  MoE:   {comparison_results['moe']['avg_ms']:.2f} ms")
    print(f"  Dense: {comparison_results['dense']['avg_ms']:.2f} ms")
    print(f"  Ratio: {comparison_results['ratio']:.2f}x")
    
    return scaling_results, comparison_results


if __name__ == "__main__":
    main()
