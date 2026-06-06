# benchmarks/benchmark_moe.py
"""Micro-benchmark for MoE routing and computation."""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe import DeepSeekMoE, AuxLossFreeGate


def benchmark_moe_routing(
    dim: int = 2048,
    n_routed_experts: int = 64,
    n_activated_experts: int = 4,
    n_shared_experts: int = 4,
    seqlen: int = 4096,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Benchmark MoE routing computation.
    
    Args:
        dim: Model dimension
        n_routed_experts: Number of routed experts
        n_activated_experts: Number of activated experts per token
        n_shared_experts: Number of shared experts
        seqlen: Sequence length
        n_warmup: Number of warmup iterations
        n_bench: Number of benchmark iterations
        device: Device to run on
        dtype: Data type
        
    Returns:
        Dictionary with benchmark results
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}
    
    # Create MoE config
    config = {
        'dim': dim,
        'n_routed_experts': n_routed_experts,
        'n_shared_experts': n_shared_experts,
        'moe_inter_dim': dim * 4,
        'n_activated_experts': n_activated_experts,
        'expert_capacity_factor': 1.5,
        'expert_dropout_prob': 0.1,
        'warmup_steps': 2000,
        'moe_activation': 'swiglu',
    }
    
    # Create MoE module
    moe = DeepSeekMoE(config).to(device).to(dtype)
    
    # Create input tensor
    x = torch.randn(seqlen, dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(n_warmup):
        y = moe(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_bench):
        y = moe(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_bench
    
    # Calculate throughput
    tokens_per_sec = seqlen / elapsed
    
    return {
        'dim': dim,
        'n_routed_experts': n_routed_experts,
        'n_activated_experts': n_activated_experts,
        'n_shared_experts': n_shared_experts,
        'seqlen': seqlen,
        'elapsed_ms': elapsed * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def benchmark_moe_vs_dense(
    dim: int = 2048,
    inter_dim: int = 8192,
    n_routed_experts: int = 64,
    n_activated_experts: int = 4,
    n_shared_experts: int = 4,
    seqlen: int = 4096,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Compare MoE vs dense FFN computation."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}
    
    # Create MoE config
    moe_config = {
        'dim': dim,
        'n_routed_experts': n_routed_experts,
        'n_shared_experts': n_shared_experts,
        'moe_inter_dim': inter_dim,
        'n_activated_experts': n_activated_experts,
        'expert_capacity_factor': 1.5,
        'expert_dropout_prob': 0.1,
        'warmup_steps': 2000,
        'moe_activation': 'swiglu',
    }
    
    # Create MoE module
    moe = DeepSeekMoE(moe_config).to(device).to(dtype)
    
    # Create dense FFN for comparison
    dense_ffn = torch.nn.Sequential(
        torch.nn.Linear(dim, inter_dim, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(inter_dim, dim, bias=False),
    ).to(device).to(dtype)
    
    # Create input tensor
    x = torch.randn(seqlen, dim, device=device, dtype=dtype)
    
    # Benchmark MoE
    for _ in range(n_warmup):
        y_moe = moe(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_bench):
        y_moe = moe(x)
    torch.cuda.synchronize()
    moe_time = (time.time() - start) / n_bench
    
    # Benchmark dense FFN
    for _ in range(n_warmup):
        y_dense = dense_ffn(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_bench):
        y_dense = dense_ffn(x)
    torch.cuda.synchronize()
    dense_time = (time.time() - start) / n_bench
    
    return {
        'dim': dim,
        'inter_dim': inter_dim,
        'n_routed_experts': n_routed_experts,
        'n_activated_experts': n_activated_experts,
        'seqlen': seqlen,
        'moe_ms': moe_time * 1000,
        'dense_ms': dense_time * 1000,
        'moe_vs_dense': dense_time / moe_time,
    }


def benchmark_routing_overhead(
    dim: int = 2048,
    n_routed_experts: int = 64,
    n_activated_experts: int = 4,
    seqlen: int = 4096,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Benchmark routing overhead only (without expert computation)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}
    
    # Create gate
    config = {
        'dim': dim,
        'n_routed_experts': n_routed_experts,
        'n_activated_experts': n_activated_experts,
    }
    gate = AuxLossFreeGate(config).to(device).to(dtype)
    
    # Create input tensor
    x = torch.randn(seqlen, dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(n_warmup):
        weights, indices = gate(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_bench):
        weights, indices = gate(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_bench
    
    return {
        'dim': dim,
        'n_routed_experts': n_routed_experts,
        'n_activated_experts': n_activated_experts,
        'seqlen': seqlen,
        'routing_overhead_ms': elapsed * 1000,
        'routing_throughput': seqlen / elapsed,
    }


def main():
    """Run benchmarks."""
    print("MoE Benchmark")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {'dim': 1024, 'n_routed_experts': 32, 'n_activated_experts': 4},
        {'dim': 2048, 'n_routed_experts': 64, 'n_activated_experts': 4},
        {'dim': 4096, 'n_routed_experts': 128, 'n_activated_experts': 8},
    ]
    
    for config in configs:
        print(f"\nBenchmarking config: {config}")
        result = benchmark_moe_routing(**config)
        if result:
            print(f"  Time: {result['elapsed_ms']:.2f} ms")
            print(f"  Throughput: {result['tokens_per_sec']:.2f} tokens/sec")
    
    # Compare MoE vs dense
    print("\n" + "=" * 50)
    print("MoE vs Dense FFN Comparison")
    print("=" * 50)
    
    for config in configs:
        print(f"\nComparing with config: {config}")
        result = benchmark_moe_vs_dense(**config)
        if result:
            print(f"  MoE: {result['moe_ms']:.2f} ms")
            print(f"  Dense: {result['dense_ms']:.2f} ms")
            print(f"  MoE vs Dense: {result['moe_vs_dense']:.2f}x")
    
    # Benchmark routing overhead
    print("\n" + "=" * 50)
    print("Routing Overhead")
    print("=" * 50)
    
    for config in configs:
        print(f"\nBenchmarking routing with config: {config}")
        result = benchmark_routing_overhead(**config)
        if result:
            print(f"  Routing overhead: {result['routing_overhead_ms']:.2f} ms")
            print(f"  Routing throughput: {result['routing_throughput']:.2f} tokens/sec")


if __name__ == "__main__":
    main()
