# benchmarks/benchmark_training.py
"""End-to-end training benchmark."""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory_profiler import get_profiler


def benchmark_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 2,
    seqlen: int = 4096,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Benchmark a full training step (forward + backward + optimizer step).
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        batch_size: Batch size
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
    
    profiler = get_profiler()
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seqlen), device=device)
    labels = torch.randint(0, 1000, (batch_size, seqlen), device=device)
    
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    
    # Benchmark
    with profiler.profile("training_step"):
        start = time.time()
        for _ in range(n_bench):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / n_bench
    
    # Calculate metrics
    tokens_per_sec = (batch_size * seqlen) / elapsed
    mfu = tokens_per_sec * sum(p.numel() for p in model.parameters()) * 6 / (312e12)  # Assuming A100
    
    return {
        'batch_size': batch_size,
        'seqlen': seqlen,
        'elapsed_ms': elapsed * 1000,
        'tokens_per_sec': tokens_per_sec,
        'mfu': mfu,
        'loss': loss.item(),
    }


def benchmark_memory_usage(
    model: torch.nn.Module,
    batch_size: int = 2,
    seqlen: int = 4096,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """Benchmark memory usage during training."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {}
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seqlen), device=device)
    labels = torch.randint(0, 1000, (batch_size, seqlen), device=device)
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Forward pass
    logits = model(input_ids)
    forward_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    # Backward pass
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    loss.backward()
    backward_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'batch_size': batch_size,
        'seqlen': seqlen,
        'forward_memory_gb': forward_mem,
        'backward_memory_gb': backward_mem,
        'total_memory_gb': backward_mem,
    }


def main():
    """Run benchmarks."""
    print("Training Benchmark")
    print("=" * 50)
    
    # This is a placeholder - actual implementation would require
    # creating a model instance with the correct config
    print("Note: This benchmark requires a model instance.")
    print("Please run with an actual model to get meaningful results.")
    
    # Example usage (commented out):
    # from models.transformer import Transformer
    # from configs import load_config
    # 
    # config = load_config("configs/pretrain.yaml")
    # model = Transformer(config['model']).to('cuda')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 
    # result = benchmark_training_step(model, optimizer)
    # print(f"Training step: {result['elapsed_ms']:.2f} ms")
    # print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
    # print(f"MFU: {result['mfu']:.2%}")
    
    print("\n" + "=" * 50)
    print("Memory Usage Benchmark")
    print("=" * 50)
    
    # This would also require a model instance
    print("Note: Memory benchmark requires a model instance.")


if __name__ == "__main__":
    main()
