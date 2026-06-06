"""Unit tests for the fused Linear + ReLU² kernel (Phase 4.5)."""

from __future__ import annotations

import pytest
import torch

from kernels.linear_relu2 import fused_linear_relu2, linear_relu2


class TestLinearRelu2:
    def test_output_shape(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        out = linear_relu2(x, w)
        assert out.shape == (4, 8)

    def test_output_non_negative(self):
        x = torch.randn(10, 32)
        w = torch.randn(16, 32)
        out = linear_relu2(x, w)
        assert (out >= 0).all()

    def test_negative_logits_become_zero(self):
        x = torch.ones(1, 4)
        w = -torch.ones(2, 4) * 10
        out = linear_relu2(x, w)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_with_bias(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        b = torch.randn(8)
        out = linear_relu2(x, w, b)
        assert out.shape == (4, 8)

    def test_squared_output(self):
        x = torch.ones(1, 4)
        w = torch.ones(2, 4) * 2
        out = linear_relu2(x, w)
        # relu(x @ W.T) = relu([8, 8]) = [8, 8]
        # square = [64, 64]
        expected = torch.tensor([[64.0, 64.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flow(self):
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(8, 16, requires_grad=True)
        out = linear_relu2(x, w)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert w.grad is not None


class TestFusedLinearRelu2:
    def test_fallback_matches_cpu(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        ref = linear_relu2(x, w)
        fused = fused_linear_relu2(x, w)
        assert torch.allclose(ref, fused, atol=1e-5)

    def test_fallback_with_bias_matches(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        b = torch.randn(8)
        ref = linear_relu2(x, w, b)
        fused = fused_linear_relu2(x, w, b)
        assert torch.allclose(ref, fused, atol=1e-5)

    def test_negative_weights_zero_output(self):
        x = torch.ones(1, 4)
        w = -torch.ones(2, 4) * 10
        ref = linear_relu2(x, w)
        fused = fused_linear_relu2(x, w)
        assert torch.allclose(ref, fused, atol=1e-5)
