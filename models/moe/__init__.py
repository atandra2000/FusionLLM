# models/moe/__init__.py
"""Mixture-of-Experts sub-modules.

Re-exports the public API for backward compatibility so that
``from models.moe import DeepSeekMoE`` continues to work.
"""

from .routing import AuxLossFreeGate, compute_routing_segments
from .experts import Expert, expert_forward_single
from .dispatch import scatter_gather_dispatch, try_grouped_gemm, all_to_all_dispatch
from .moe import DeepSeekMoE

__all__ = [
    "AuxLossFreeGate",
    "compute_routing_segments",
    "Expert",
    "expert_forward_single",
    "scatter_gather_dispatch",
    "try_grouped_gemm",
    "all_to_all_dispatch",
    "DeepSeekMoE",
]
