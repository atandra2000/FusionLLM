# utils/memory_profiler.py
"""Memory profiler utility for benchmarking and profiling."""

import torch
from contextlib import contextmanager
from typing import Optional, Dict, List


class MemoryProfiler:
    """Memory profiler for tracking GPU memory usage during training."""
    
    def __init__(self):
        self.snapshots: List[Dict] = []
        self.enabled = True
    
    @contextmanager
    def profile(self, name: str):
        """Profile memory usage within a context manager.
        
        Args:
            name: Name of the profiling context
        """
        if not self.enabled:
            yield
            return
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            start_mem = torch.cuda.memory_allocated()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            yield
            
            end_time.record()
            torch.cuda.synchronize()
            
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            
            # Calculate timing
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            # For CPU or MPS, use time-based profiling
            import time
            start_time_event = time.time()
            start_mem = 0.0  # Can't measure memory on non-CUDA devices
            
            yield
            
            elapsed_time = time.time() - start_time_event
            end_mem = 0.0
            peak_mem = 0.0
        
        self.snapshots.append({
            'name': name,
            'start': start_mem / 1024**3,  # Convert to GB
            'end': end_mem / 1024**3,
            'peak': peak_mem / 1024**3,
            'delta': (end_mem - start_mem) / 1024**3,
            'elapsed': elapsed_time,
        })
    
    def report(self) -> str:
        """Generate a report of all profiling snapshots."""
        lines = ["Memory Profile Report", "=" * 50]
        
        total_start = 0
        total_end = 0
        total_peak = 0
        total_delta = 0
        total_time = 0
        
        for snap in self.snapshots:
            lines.append(f"\n{snap['name']}:")
            lines.append(f"  Start:     {snap['start']:.3f} GB")
            lines.append(f"  End:       {snap['end']:.3f} GB")
            lines.append(f"  Peak:      {snap['peak']:.3f} GB")
            lines.append(f"  Delta:     {snap['delta']:.3f} GB")
            lines.append(f"  Time:      {snap['elapsed']:.3f} s")
            
            total_start += snap['start']
            total_end += snap['end']
            total_peak += snap['peak']
            total_delta += snap['delta']
            total_time += snap['elapsed']
        
        if self.snapshots:
            lines.append(f"\n{'=' * 50}")
            lines.append(f"Summary:")
            lines.append(f"  Total snapshots: {len(self.snapshots)}")
            lines.append(f"  Total time:      {total_time:.3f} s")
            lines.append(f"  Avg delta:       {total_delta / len(self.snapshots):.3f} GB/snapshot")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all snapshots."""
        self.snapshots.clear()
    
    def get_summary(self) -> Dict:
        """Get a summary dictionary of memory usage."""
        if not self.snapshots:
            return {}
        
        return {
            'num_snapshots': len(self.snapshots),
            'total_time_s': sum(s['elapsed'] for s in self.snapshots),
            'peak_memory_gb': max(s['peak'] for s in self.snapshots),
            'avg_delta_gb': sum(s['delta'] for s in self.snapshots) / len(self.snapshots),
        }


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information."""
    if not torch.cuda.is_available():
        # Try MPS (Apple Silicon)
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_device'):
            try:
                return {
                    'allocated_gb': torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0.0,
                    'reserved_gb': 0.0,  # MPS doesn't have reserved memory
                    'max_allocated_gb': torch.mps.max_memory_allocated() / 1024**3 if hasattr(torch.mps, 'max_memory_allocated') else 0.0,
                    'max_reserved_gb': 0.0,
                }
            except:
                return {}
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
    }


def estimate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """Estimate model memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory (assuming fp32)
    param_memory_gb = total_params * 4 / 1024**3  # 4 bytes per param
    grad_memory_gb = trainable_params * 4 / 1024**3
    optimizer_memory_gb = trainable_params * 8 / 1024**3  # AdamW uses 2x params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_memory_gb': param_memory_gb,
        'grad_memory_gb': grad_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'total_memory_gb': param_memory_gb + grad_memory_gb + optimizer_memory_gb,
    }


# Global memory profiler instance
_global_profiler: Optional[MemoryProfiler] = None


def get_profiler() -> MemoryProfiler:
    """Get the global memory profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MemoryProfiler()
    return _global_profiler


def profile_context(name: str):
    """Convenience function to get a profile context from the global profiler."""
    return get_profiler().profile(name)
