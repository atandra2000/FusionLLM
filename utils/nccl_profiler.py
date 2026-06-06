# utils/nccl_profiler.py
"""NCCL profiling utilities for FusionLLM.

Provides profiling tools for distributed communication:
- NCCL version detection
- Communication pattern analysis
- Latency measurement
- Bandwidth utilization metrics
"""

import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
import time


@dataclass
class NCCLProfileConfig:
    """Configuration for NCCL profiling."""
    enable_profiling: bool = True
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    sync_device: bool = True
    record_shapes: bool = True


@dataclass
class NCCLProfileResult:
    """Result of NCCL profiling."""
    operation: str
    latency_ms: float
    bandwidth_gbps: float
    message_size_bytes: int
    world_size: int
    rank: int


class NCCLProfiler:
    """NCCL communication profiler."""
    
    def __init__(self, config: Optional[NCCLProfileConfig] = None):
        self.config = config or NCCLProfileConfig()
        self.results: List[NCCLProfileResult] = []
        self.is_distributed = dist.is_initialized()
        
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    @contextmanager
    def profile_operation(self, operation: str, message_size: int = 0):
        """Profile a communication operation.
        
        Args:
            operation: Name of the operation
            message_size: Size of message in bytes
        """
        if not self.config.enable_profiling or not self.is_distributed:
            yield
            return
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            if self.config.sync_device:
                torch.cuda.synchronize()
            dist.barrier()
        
        # Benchmark
        latencies = []
        for _ in range(self.config.benchmark_iterations):
            if self.config.sync_device:
                torch.cuda.synchronize()
            
            start = time.time()
            yield
            if self.config.sync_device:
                torch.cuda.synchronize()
            end = time.time()
            
            latencies.append((end - start) * 1000)  # Convert to ms
            dist.barrier()
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        bandwidth = (message_size * 8) / (avg_latency * 1e6) if avg_latency > 0 else 0  # Gbps
        
        result = NCCLProfileResult(
            operation=operation,
            latency_ms=avg_latency,
            bandwidth_gbps=bandwidth,
            message_size_bytes=message_size,
            world_size=self.world_size,
            rank=self.rank,
        )
        
        self.results.append(result)
    
    def profile_all_reduce(self, tensor: torch.Tensor) -> NCCLProfileResult:
        """Profile all-reduce operation."""
        message_size = tensor.nelement() * tensor.element_size()
        
        with self.profile_operation("all_reduce", message_size):
            dist.all_reduce(tensor)
        
        return self.results[-1]
    
    def profile_all_gather(self, tensor: torch.Tensor) -> NCCLProfileResult:
        """Profile all-gather operation."""
        message_size = tensor.nelement() * tensor.element_size()
        
        gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        
        with self.profile_operation("all_gather", message_size):
            dist.all_gather(gather_list, tensor)
        
        return self.results[-1]
    
    def profile_reduce_scatter(self, tensor: torch.Tensor) -> NCCLProfileResult:
        """Profile reduce-scatter operation."""
        message_size = tensor.nelement() * tensor.element_size()
        
        output = torch.zeros_like(tensor)
        
        with self.profile_operation("reduce_scatter", message_size):
            dist.reduce_scatter(output, [tensor for _ in range(self.world_size)])
        
        return self.results[-1]
    
    def profile_broadcast(self, tensor: torch.Tensor, src: int = 0) -> NCCLProfileResult:
        """Profile broadcast operation."""
        message_size = tensor.nelement() * tensor.element_size()
        
        with self.profile_operation("broadcast", message_size):
            dist.broadcast(tensor, src)
        
        return self.results[-1]
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of profiling results."""
        summary = {}
        
        for result in self.results:
            if result.operation not in summary:
                summary[result.operation] = {
                    'count': 0,
                    'total_latency_ms': 0,
                    'total_bandwidth_gbps': 0,
                    'avg_latency_ms': 0,
                    'avg_bandwidth_gbps': 0,
                }
            
            summary[result.operation]['count'] += 1
            summary[result.operation]['total_latency_ms'] += result.latency_ms
            summary[result.operation]['total_bandwidth_gbps'] += result.bandwidth_gbps
        
        # Calculate averages
        for op in summary:
            count = summary[op]['count']
            if count > 0:
                summary[op]['avg_latency_ms'] = summary[op]['total_latency_ms'] / count
                summary[op]['avg_bandwidth_gbps'] = summary[op]['total_bandwidth_gbps'] / count
        
        return summary
    
    def print_summary(self):
        """Print summary of profiling results."""
        if not self.results:
            print("No profiling results available")
            return
        
        print(f"\n{'='*60}")
        print(f"NCCL Profiling Summary")
        print(f"{'='*60}")
        print(f"World Size: {self.world_size}")
        print(f"Rank: {self.rank}")
        print(f"{'='*60}\n")
        
        summary = self.get_summary()
        
        for op, stats in summary.items():
            print(f"{op}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.3f} ms")
            print(f"  Avg Bandwidth: {stats['avg_bandwidth_gbps']:.2f} Gbps")
            print()
    
    def reset(self):
        """Reset profiling results."""
        self.results.clear()


def get_nccl_version() -> str:
    """Get NCCL version string."""
    try:
        import NCCL
        return NCCL.__version__
    except ImportError:
        return "Unknown (NCCL not available)"


def check_nccl_available() -> bool:
    """Check if NCCL is available."""
    try:
        import NCCL
        return True
    except ImportError:
        return False


def profile_communication(
    model: torch.nn.Module,
    input_shape: tuple = (1, 64),
    device: str = 'cuda',
    config: Optional[NCCLProfileConfig] = None,
) -> Dict[str, Dict]:
    """Profile communication patterns in a model.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Device to run on
        config: Profiling configuration
        
    Returns:
        Dictionary with profiling results
    """
    if not dist.is_initialized():
        print("Distributed not initialized, skipping communication profiling")
        return {}
    
    profiler = NCCLProfiler(config)
    
    # Create dummy tensor for profiling
    tensor = torch.randn(input_shape, device=device)
    
    # Profile different operations
    profiler.profile_all_reduce(tensor.clone())
    profiler.profile_all_gather(tensor.clone())
    profiler.profile_reduce_scatter(tensor.clone())
    profiler.profile_broadcast(tensor.clone())
    
    profiler.print_summary()
    
    return profiler.get_summary()
