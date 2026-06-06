# models/moe/moe.py
"""DeepSeekMoE: Mixture-of-Experts with shared experts and aux-loss-free load balancing.

This is the main orchestrator that composes routing, expert computation,
and dispatch strategies.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.moe import dispatch as _dispatch
from models.moe.experts import Expert, expert_forward_single
from models.moe.routing import AuxLossFreeGate


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with shared experts and aux-loss-free load balancing.

    Expert parallelism
    ------------------
    Each rank owns a contiguous shard of n_local_experts routed experts
    (indices [experts_start, experts_end)).  Shared experts are replicated
    on all ranks and run unconditionally.

    All-reduce
    ------------
    Only the routed expert output (y_routed) is all-reduced across ranks.
    Shared expert outputs are computed locally on every rank and added AFTER
    the all-reduce, so they are never multiplied by world_size.

    Expert capacity factor
    ----------------------
    A capacity factor limits the maximum number of tokens that can be processed
    by a single expert, preventing token pile-up on popular experts during
    early training.  The capacity is computed as:
        capacity = capacity_factor * (total_assignments / n_routed_experts)
    Tokens beyond this capacity are dropped (their contribution set to zero).

    Expert dropout
    --------------
    During training, a small fraction (default 0.1) of experts is randomly
    skipped, forcing the remaining experts to diversify and reducing over-
    specialisation.  Dropped experts receive no gradient and their bias is
    not updated for that step.

    Routing cache
    -------------
    The most recent (weights, indices) pair is stored in self._last_weights and self._last_indices
    after every forward pass. This allows get_load_balance_loss() and get_routing_stats() to reuse routing
    without a second gate call, and allows pretrain.py to call update_gate_bias() without re-embedding the input batch.
    """

    def __init__(
        self, config: dict, world_size: int = 1, rank: int = 0, tp_size: int = 1, tp_rank: int = 0
    ):
        super().__init__()
        self.dim = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.moe_inter_dim = config["moe_inter_dim"]
        self.world_size = world_size
        self.rank = rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.capacity_factor = config.get("expert_capacity_factor", 1.5)
        self.expert_dropout_prob = config.get("expert_dropout_prob", 0.1)
        self.warmup_steps = config.get("warmup_steps", 2000)
        self.use_triton_grouped_gemm = bool(config.get("use_triton_grouped_gemm", False))
        self.use_all_to_all = bool(config.get("use_all_to_all_dispatch", False))
        self._train_steps = 0

        if self.n_routed_experts % world_size != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be "
                f"divisible by world_size ({world_size})"
            )
        self.n_local_experts = self.n_routed_experts // world_size
        self.experts_start = rank * self.n_local_experts
        self.experts_end = self.experts_start + self.n_local_experts

        self.activation = config.get("moe_activation", "swiglu")

        self.gate = AuxLossFreeGate(config)

        self.experts = nn.ModuleList(
            [
                Expert(
                    self.dim,
                    self.moe_inter_dim,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    activation=self.activation,
                )
                for _ in range(self.n_local_experts)
            ]
        )

        self.shared_experts = nn.ModuleList(
            [
                Expert(
                    self.dim,
                    self.moe_inter_dim,
                    tp_size=1,
                    tp_rank=0,
                    activation=self.activation,
                )
                for _ in range(self.n_shared_experts)
            ]
        )

        # Routing cache: populated during forward(), reused by auxiliary methods
        self._last_weights: torch.Tensor | None = None
        self._last_indices: torch.Tensor | None = None

        # Precompute expert weight stacks to avoid per-forward torch.stack
        self._expert_w1_stack: torch.Tensor | None = None
        self._expert_w2_stack: torch.Tensor | None = None
        self._expert_w3_stack: torch.Tensor | None = None
        self._shared_w1_stack: torch.Tensor | None = None
        self._shared_w2_stack: torch.Tensor | None = None
        self._shared_w3_stack: torch.Tensor | None = None

        self._refresh_weight_stacks()

    def _refresh_weight_stacks(self) -> None:
        """Refresh precomputed weight stacks after optimizer step."""
        if len(self.experts) > 0:
            self._expert_w1_stack = torch.stack([e.w1.weight for e in self.experts])
            self._expert_w2_stack = torch.stack([e.w2.weight for e in self.experts])
            if self.activation == "swiglu":
                self._expert_w3_stack = torch.stack([e.w3.weight for e in self.experts])
            else:
                self._expert_w3_stack = None
        else:
            self._expert_w1_stack = None
            self._expert_w2_stack = None
            self._expert_w3_stack = None

        if self.shared_experts:
            self._shared_w1_stack = torch.stack([e.w1.weight for e in self.shared_experts])
            self._shared_w2_stack = torch.stack([e.w2.weight for e in self.shared_experts])
            if self.activation == "swiglu":
                self._shared_w3_stack = torch.stack([e.w3.weight for e in self.shared_experts])
            else:
                self._shared_w3_stack = None
        else:
            self._shared_w1_stack = None
            self._shared_w2_stack = None
            self._shared_w3_stack = None

    def _expert_forward_single(
        self, x_group: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor | None
    ) -> torch.Tensor:
        """Single expert forward pass (SwiGLU or ReLU²)."""
        return expert_forward_single(x_group, w1, w2, w3, self.activation)

    def _compute_shared_experts(self, flat: torch.Tensor) -> torch.Tensor:
        """Compute all shared expert outputs and sum them."""
        if not self.shared_experts:
            return torch.zeros_like(flat)
        if self._shared_w1_stack is not None:
            out = torch.zeros_like(flat)
            for i in range(len(self.shared_experts)):
                out = out + self._expert_forward_single(
                    flat, self._shared_w1_stack[i], self._shared_w2_stack[i],
                    self._shared_w3_stack[i] if self._shared_w3_stack is not None else None,
                )
            return out
        return torch.stack([e(flat) for e in self.shared_experts], dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, dim) — flattened token representations
        Returns:
            Tensor of same shape as x.
        """
        shape = x.shape
        flat = x.view(-1, self.dim)
        T = flat.size(0)

        # ── Routing ────────────────────────────────────────────────────────
        topk = self.gate.topk
        weights, indices = self.gate(flat)
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()

        # ── Expert dropout (training only) ────────────────────────────────
        if (
            self.training
            and self.expert_dropout_prob > 0
            and self._train_steps <= self.warmup_steps
        ):
            dropout_mask = (
                torch.rand(self.n_local_experts, device=flat.device) > self.expert_dropout_prob
            )
            if not dropout_mask.any():
                dropout_mask[0] = True
        else:
            dropout_mask = torch.ones(self.n_local_experts, dtype=torch.bool, device=flat.device)

        # ── Scatter-gather routing ─────────────────────────────────────────
        topk = weights.size(1)
        y_routed = torch.zeros_like(flat)

        total_assign = T * topk
        flat_expert_global = indices.flatten()
        flat_weights = weights.flatten()
        flat_token_ids = torch.arange(T, device=flat.device).repeat_interleave(topk)

        sort_idx = flat_expert_global.argsort()
        flat_expert_global_sorted = flat_expert_global[sort_idx]
        flat_token_ids_sorted = flat_token_ids[sort_idx]
        flat_weights_sorted = flat_weights[sort_idx]

        unique_experts, segment_sizes = torch.unique_consecutive(
            flat_expert_global_sorted, return_counts=True
        )
        segment_starts = torch.cat(
            [torch.tensor([0], device=flat.device), segment_sizes.cumsum(0)[:-1]]
        )

        expert_start = torch.full((self.n_local_experts,), -1, dtype=torch.long, device=flat.device)
        expert_size = torch.zeros(self.n_local_experts, dtype=torch.long, device=flat.device)

        valid_mask = (unique_experts >= self.experts_start) & (
            unique_experts < self.experts_start + self.n_local_experts
        )
        local_expert_ids = unique_experts[valid_mask] - self.experts_start
        expert_start[local_expert_ids] = segment_starts[valid_mask]
        expert_size[local_expert_ids] = segment_sizes[valid_mask]

        expected_assign = (total_assign + self.n_routed_experts - 1) // self.n_routed_experts
        capacity = int(self.capacity_factor * expected_assign)
        expert_size = torch.minimum(
            expert_size, torch.tensor(capacity, dtype=torch.long, device=flat.device)
        )

        active_mask = dropout_mask & (expert_start >= 0) & (expert_size > 0)
        active_indices = torch.where(active_mask)[0]
        active_list = active_indices.tolist()

        # ── Dispatch ──────────────────────────────────────────────────────
        if self.use_all_to_all and self.world_size > 1 and dist.is_initialized() and len(active_list) > 0:
            y_routed = _dispatch.all_to_all_dispatch(
                flat, flat_token_ids_sorted, flat_weights_sorted,
                expert_start, expert_size, active_indices, y_routed,
                self._expert_w1_stack, self._expert_w2_stack, self._expert_w3_stack,
                self.activation, self.dim, self.world_size,
                self._expert_forward_single,
            )
        else:
            scatter_gather_needed = True
            if len(active_list) > 0 and _dispatch.try_grouped_gemm(
                flat, flat_token_ids_sorted, flat_weights_sorted,
                expert_start, expert_size, active_indices, y_routed,
                self._expert_w1_stack, self._expert_w2_stack, self._expert_w3_stack,
                self.activation,
            ):
                scatter_gather_needed = False

            if scatter_gather_needed and len(active_list) > 0:
                _dispatch.scatter_gather_dispatch(
                    flat, flat_token_ids_sorted, flat_weights_sorted,
                    expert_start, expert_size, active_indices, y_routed,
                    self._expert_w1_stack, self._expert_w2_stack, self._expert_w3_stack,
                    self.activation, self.dim, self._expert_forward_single,
                )

            if self.world_size > 1 and dist.is_initialized():
                dist.all_reduce(y_routed, op=dist.ReduceOp.SUM)

        # ── Shared experts (always executed, added after all-reduce) ───────
        y = y_routed + self._compute_shared_experts(flat)

        if self.training:
            self._train_steps += 1

        return y.view(shape)

    # ──────────────────────────────────────────────────────────────────────
    # Auxiliary methods (reuse cached routing — no second gate call)
    # ──────────────────────────────────────────────────────────────────────

    def _get_weighted_onehot(self) -> torch.Tensor:
        """Build (T*topk, E) one-hot assignment matrix weighted by routing scores."""
        if self._last_weights is None or self._last_indices is None:
            return torch.empty(0, self.n_routed_experts, device=self.gate.weight.device)
        one_hot = F.one_hot(self._last_indices.flatten(), num_classes=self.n_routed_experts).float()
        return one_hot * self._last_weights.flatten().unsqueeze(-1)

    def get_load_balance_loss(self) -> torch.Tensor:
        weighted_onehot = self._get_weighted_onehot()
        if weighted_onehot.numel() == 0:
            return self.gate._zero.to(self.gate.weight.device)
        T = self._last_weights.size(0)
        counts = torch.bincount(
            self._last_indices.flatten(), minlength=self.n_routed_experts
        ).float()
        f = counts / counts.sum().clamp(min=1e-10)
        P = weighted_onehot.view(T, -1, self.n_routed_experts).sum(dim=1).mean(dim=0)
        return (f * P).sum() * self.n_routed_experts

    def get_z_loss(self) -> torch.Tensor:
        """Router z-loss from the gate's cached pre-sigmoid logits."""
        return self.gate.get_z_loss()

    def get_routing_stats(self) -> dict[str, torch.Tensor]:
        weighted_onehot = self._get_weighted_onehot()
        if weighted_onehot.numel() == 0:
            return {}
        E = self.n_routed_experts
        counts = torch.bincount(self._last_indices.flatten(), minlength=E).float()
        load = counts / counts.sum().clamp(min=1e-10)
        weight_sum = weighted_onehot.sum(dim=0)
        mean_weight = weight_sum / counts.clamp(min=1.0)
        utilisation = (counts > 0).float().mean()
        return {
            "counts": counts,
            "load": load,
            "mean_weight": mean_weight,
            "utilisation": utilisation,
        }

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """
        Update the gate's load-balancing bias using the cached token counts.
        """
        if self._last_indices is None:
            return
        counts = torch.bincount(
            self._last_indices.flatten().cpu(),
            minlength=self.n_routed_experts,
        )
        self.gate.update_bias(counts, speed=speed)

    # ──────────────────────────────────────────────────────────────────────
    # Backward-compatible method wrappers
    # ──────────────────────────────────────────────────────────────────────

    def _try_grouped_gemm(self, flat, flat_token_ids_sorted, flat_weights_sorted,
                          expert_start, expert_size, active_indices, y_routed):
        """Backward-compatible wrapper for dispatch.try_grouped_gemm."""
        return _dispatch.try_grouped_gemm(
            flat, flat_token_ids_sorted, flat_weights_sorted,
            expert_start, expert_size, active_indices, y_routed,
            self._expert_w1_stack, self._expert_w2_stack, self._expert_w3_stack,
            self.activation,
        )

    def _all_to_all_dispatch(self, flat, flat_token_ids_sorted, flat_weights_sorted,
                             expert_start, expert_size, active_indices, y_routed, dropout_mask):
        """Backward-compatible wrapper for dispatch.all_to_all_dispatch."""
        return _dispatch.all_to_all_dispatch(
            flat, flat_token_ids_sorted, flat_weights_sorted,
            expert_start, expert_size, active_indices, y_routed,
            self._expert_w1_stack, self._expert_w2_stack, self._expert_w3_stack,
            self.activation, self.dim, self.world_size,
            self._expert_forward_single,
        )
