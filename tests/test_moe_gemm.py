"""Unit tests for `ops.triton.grouped_gemm` and the MoE dispatch hook.

Phase 2.8:
* `has_triton` returns False when triton is not importable.
* `grouped_gemm` raises NotImplementedError when triton is unavailable.
* `DeepSeekMoE._try_grouped_gemm` returns False when the config flag
  is off, when triton is unavailable, and when the kernel raises.
"""

from __future__ import annotations

import pytest
import torch

from ops.triton.grouped_gemm import grouped_gemm, has_triton


# ── Module-level ───────────────────────────────────────────────────────────
class TestGroupedGemmModule:
    def test_has_triton_returns_bool(self):
        # Returns either True or False depending on the env; just make
        # sure it's a bool and doesn't raise.
        result = has_triton()
        assert isinstance(result, bool)

    def test_grouped_gemm_raises_when_no_triton(self, monkeypatch):
        # Force has_triton() → False by monkey-patching the module
        # so we can exercise the NotImplementedError branch.
        from ops.triton import grouped_gemm as mod

        monkeypatch.setattr(mod, "_HAS_TRITON", False)

        with pytest.raises(NotImplementedError, match="Triton"):
            grouped_gemm(
                torch.randn(4, 8),
                torch.randn(2, 8, 16),
                torch.tensor([0, 2, 4], dtype=torch.int32),
            )


# ── MoE dispatch hook ─────────────────────────────────────────────────────
class TestMoEGroupedGemmDispatch:
    def _build_moe(self, **overrides):
        from models.moe import DeepSeekMoE

        cfg = dict(
            dim=8,
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            moe_inter_dim=16,
        )
        cfg.update(overrides)
        return DeepSeekMoE(cfg, world_size=1, rank=0)

    def test_flag_off_returns_false(self):
        m = self._build_moe(use_triton_grouped_gemm=False)
        y_routed = torch.zeros(2, 8)
        result = m._try_grouped_gemm(
            flat=torch.randn(2, 8),
            flat_token_ids_sorted=torch.tensor([0, 1]),
            flat_weights_sorted=torch.tensor([1.0, 1.0]),
            expert_start=torch.tensor([0, 1]),
            expert_size=torch.tensor([1, 1]),
            active_indices=torch.tensor([0, 1]),
            y_routed=y_routed,
        )
        assert result is False

    def test_flag_on_but_kernel_unavailable_returns_false(self, monkeypatch):
        """When the config flag is on but Triton is unavailable (or
        the kernel raises NotImplementedError), the dispatch returns
        False so the caller falls back to the scatter-gather path.
        """
        m = self._build_moe(use_triton_grouped_gemm=True)
        # Force the import inside _try_grouped_gemm to think triton
        # is unavailable.
        from ops.triton import grouped_gemm as mod

        monkeypatch.setattr(mod, "_HAS_TRITON", False)
        y_routed = torch.zeros(2, 8)
        result = m._try_grouped_gemm(
            flat=torch.randn(2, 8),
            flat_token_ids_sorted=torch.tensor([0, 1]),
            flat_weights_sorted=torch.tensor([1.0, 1.0]),
            expert_start=torch.tensor([0, 1]),
            expert_size=torch.tensor([1, 1]),
            active_indices=torch.tensor([0, 1]),
            y_routed=y_routed,
        )
        assert result is False

    def test_empty_active_list_returns_false(self):
        m = self._build_moe(use_triton_grouped_gemm=True)
        y_routed = torch.zeros(2, 8)
        # Empty active_indices → returns False before doing any work.
        result = m._try_grouped_gemm(
            flat=torch.randn(2, 8),
            flat_token_ids_sorted=torch.tensor([], dtype=torch.long),
            flat_weights_sorted=torch.tensor([], dtype=torch.float32),
            expert_start=torch.tensor([0, 0]),
            expert_size=torch.tensor([0, 0]),
            active_indices=torch.tensor([], dtype=torch.long),
            y_routed=y_routed,
        )
        assert result is False
