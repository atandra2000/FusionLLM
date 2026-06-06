# benchmarks/__init__.py
"""Benchmark utilities for FusionLLM."""

from .benchmark_delta_rule import benchmark_delta_rule, benchmark_delta_rule_vs_sequential
from .benchmark_moe import benchmark_moe_routing, benchmark_moe_vs_dense, benchmark_routing_overhead
from .benchmark_training import benchmark_training_step, benchmark_memory_usage

__all__ = [
    'benchmark_delta_rule',
    'benchmark_delta_rule_vs_sequential',
    'benchmark_moe_routing',
    'benchmark_moe_vs_dense',
    'benchmark_routing_overhead',
    'benchmark_training_step',
    'benchmark_memory_usage',
]
