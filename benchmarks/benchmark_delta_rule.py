# benchmarks/benchmark_delta_rule.py
"""Micro-benchmark for GatedDeltaNet delta-rule implementation."""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.delta_rule import chunked_delta_rule


def benchmark_delta_rule(
    seqlen: int,
    bsz: int = 2,
    n_heads: int = 4,
    headdim: int = 64,
    d_state: int = 64,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Benchmark GatedDeltaNet delta-rule implementation.
    
    Args:
        seqlen: Sequence length
        bsz: Batch size
        n_heads: Number of attention heads
        headdim: Head dimension
        d_state: State dimension
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
    
    # Create input tensors
    v = torch.randn(bsz, seqlen, n_heads, headdim, device=device, dtype=dtype)
    dt = torch.randn(bsz, seqlen, n_heads, device=device, dtype=dtype)
    A = -torch.arange(1, n_heads + 1, device=device, dtype=dtype).unsqueeze(-1)
    B = torch.randn(bsz, seqlen, n_heads, d_state, device=device, dtype=dtype)
    C = torch.randn(bsz, seqlen, n_heads, d_state, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(n_warmup):
        try:
            y = chunked_delta_rule(v.float(), dt.float(), A.float(), B.float(), C.float())
        except Exception as e:
            print(f"Error during warmup: {e}")
            return {}
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(n_bench):
        try:
            y = chunked_delta_rule(v.float(), dt.float(), A.float(), B.float(), C.float())
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return {}
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_bench
    
    # Calculate throughput
    tokens_per_sec = (bsz * seqlen) / elapsed
    
    return {
        'seqlen': seqlen,
        'bsz': bsz,
        'n_heads': n_heads,
        'headdim': headdim,
        'd_state': d_state,
        'elapsed_ms': elapsed * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def benchmark_delta_rule_vs_sequential(
    seqlen: int,
    bsz: int = 2,
    n_heads: int = 4,
    headdim: int = 64,
    d_state: int = 64,
    n_warmup: int = 5,
    n_bench: int = 20,
    device: str = 'cuda',
):
    """Compare chunked vs sequential delta-rule implementations.
    
    Note: Sequential implementation is not available in the current codebase.
    This benchmark only tests the chunked implementation.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}
    
    # Create input tensors
    v = torch.randn(bsz, seqlen, n_heads, headdim, device=device, dtype=torch.float32)
    dt = torch.randn(bsz, seqlen, n_heads, device=device, dtype=torch.float32)
    A = -torch.arange(1, n_heads + 1, device=device, dtype=torch.float32).unsqueeze(-1)
    B = torch.randn(bsz, seqlen, n_heads, d_state, device=device, dtype=torch.float32)
    C = torch.randn(bsz, seqlen, n_heads, d_state, device=device, dtype=torch.float32)
    
    # Benchmark chunked implementation
    for _ in range(n_warmup):
        y_chunked = chunked_delta_rule(v, dt, A, B, C)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_bench):
        y_chunked = chunked_delta_rule(v, dt, A, B, C)
    torch.cuda.synchronize()
    chunked_time = (time.time() - start) / n_bench
    
    results = {
        'seqlen': seqlen,
        'chunked_ms': chunked_time * 1000,
    }
    
    return results


def main():
    """Run benchmarks."""
    print("GatedDeltaNet Delta-Rule Benchmark")
    print("=" * 50)
    
    # Test different sequence lengths
    seqlens = [1024, 2048, 4096, 8192, 16384]
    
    for seqlen in seqlens:
        print(f"\nBenchmarking seqlen={seqlen}...")
        result = benchmark_delta_rule(seqlen=seqlen)
        if result:
            print(f"  Time: {result['elapsed_ms']:.2f} ms")
            print(f"  Throughput: {result['tokens_per_sec']:.2f} tokens/sec")
    
    # Compare chunked vs sequential
    print("\n" + "=" * 50)
    print("Chunked vs Sequential Comparison")
    print("=" * 50)
    
    for seqlen in [1024, 4096, 8192]:
        print(f"\nComparing at seqlen={seqlen}...")
        result = benchmark_delta_rule_vs_sequential(seqlen=seqlen)
        if result:
            print(f"  Chunked: {result['chunked_ms']:.2f} ms")
            if 'sequential_ms' in result:
                print(f"  Sequential: {result['sequential_ms']:.2f} ms")
                print(f"  Speedup: {result['speedup']:.1f}x")


if __name__ == "__main__":
    main()
