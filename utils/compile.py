# utils/compile.py
"""torch.compile utilities for FusionLLM.

Provides compilation utilities for optimizing inference and training:
- selective_compile: Compile only specific modules
- get_default_config: Get default compilation configuration
- verify_compilation: Verify compilation works correctly
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Any
from functools import wraps


def get_default_config() -> Dict[str, Any]:
    """Get default compilation configuration.
    
    Returns:
        Dictionary with compilation settings
    """
    return {
        'mode': 'max-autotune',  # 'reduce-overhead', 'max-autotune', or None
        'dynamic': True,         # Enable dynamic shapes
        'fullgraph': False,      # Allow graph breaks
        'backend': 'inductor',   # Backend ('inductor', 'cudagraphs')
        'options': {},           # Backend-specific options
    }


def selective_compile(
    model: nn.Module,
    compile_fn: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    skip_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """Compile specific modules in a model.
    
    Args:
        model: Model to compile
        compile_fn: Compilation function (default: torch.compile)
        config: Compilation configuration
        skip_patterns: Module name patterns to skip
        
    Returns:
        Model with compiled modules
    """
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile not available, skipping compilation")
        return model
    
    config = config or get_default_config()
    compile_fn = compile_fn or torch.compile
    
    skip_patterns = skip_patterns or [
        'embed', 'head',  # Embeddings
        'norm',           # RMSNorm
        'reshape', 'view', 'permute',  # Reshape ops
    ]
    
    def should_compile(name: str, module: nn.Module) -> bool:
        """Check if module should be compiled."""
        for pattern in skip_patterns:
            if pattern in name.lower():
                return False
        # Only compile leaf modules with parameters
        if len(list(module.children())) > 0:
            return False
        if sum(p.numel() for p in module.parameters()) == 0:
            return False
        return True
    
    # Compile matching modules
    compiled_count = 0
    for name, module in model.named_modules():
        if should_compile(name, module):
            try:
                compiled_module = compile_fn(
                    module,
                    mode=config.get('mode'),
                    dynamic=config.get('dynamic', True),
                    fullgraph=config.get('fullgraph', False),
                    backend=config.get('backend', 'inductor'),
                    options=config.get('options', {}),
                )
                # Replace in model
                parts = name.split('.')
                if len(parts) == 1:
                    setattr(model, parts[0], compiled_module)
                else:
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], compiled_module)
                compiled_count += 1
            except Exception as e:
                print(f"Warning: Failed to compile {name}: {e}")
    
    print(f"Compiled {compiled_count} modules")
    return model


def compile_model(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Compile the entire model.
    
    Args:
        model: Model to compile
        config: Compilation configuration
        
    Returns:
        Compiled model
    """
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile not available, skipping compilation")
        return model
    
    config = config or get_default_config()
    
    return torch.compile(
        model,
        mode=config.get('mode'),
        dynamic=config.get('dynamic', True),
        fullgraph=config.get('fullgraph', False),
        backend=config.get('backend', 'inductor'),
        options=config.get('options', {}),
    )


def verify_compilation(
    model: nn.Module,
    input_shape: tuple = (1, 64),
    device: str = 'cuda',
) -> bool:
    """Verify that compilation works correctly.
    
    Args:
        model: Model to verify
        input_shape: Input tensor shape
        device: Device to run on
        
    Returns:
        True if compilation works, False otherwise
    """
    try:
        # Create dummy input
        dummy_input = torch.randint(0, 1000, input_shape, device=device)
        
        # Test uncompiled forward
        with torch.no_grad():
            output1 = model(dummy_input)
        
        # Compile model
        compiled_model = compile_model(model)
        
        # Test compiled forward
        with torch.no_grad():
            output2 = compiled_model(dummy_input)
        
        # Verify outputs match
        if torch.allclose(output1, output2, atol=1e-4):
            print("✓ Compilation verification passed")
            return True
        else:
            print("✗ Compilation verification failed: outputs differ")
            return False
            
    except Exception as e:
        print(f"✗ Compilation verification failed: {e}")
        return False


def profile_compilation(
    model: nn.Module,
    input_shape: tuple = (1, 64),
    n_warmup: int = 3,
    n_bench: int = 10,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Profile compilation performance.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        n_warmup: Warmup iterations
        n_bench: Benchmark iterations
        device: Device to run on
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        return {}
    
    dummy_input = torch.randint(0, 1000, input_shape, device=device)
    
    # Profile uncompiled
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_bench):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        uncompiled_time = (time.time() - start) / n_bench
    
    # Profile compiled
    compiled_model = compile_model(model)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = compiled_model(dummy_input)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_bench):
            _ = compiled_model(dummy_input)
        torch.cuda.synchronize()
        compiled_time = (time.time() - start) / n_bench
    
    speedup = uncompiled_time / compiled_time if compiled_time > 0 else 0
    
    results = {
        'uncompiled_ms': uncompiled_time * 1000,
        'compiled_ms': compiled_time * 1000,
        'speedup': speedup,
    }
    
    print(f"Uncompiled: {results['uncompiled_ms']:.2f} ms")
    print(f"Compiled: {results['compiled_ms']:.2f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
    
    return results
