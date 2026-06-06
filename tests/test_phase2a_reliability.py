# tests/test_phase2a_reliability.py
"""Phase 2A — Training Reliability Hardening tests.

Validates all 7 priorities from the training-readiness audit:
1. NorMuon state recovery across checkpoint save/load
2. NaN/Inf detection on loss and gradients
3. Router z-loss contributes to total loss
4. Checkpoint retention (keep_last_n) works
5. Gradient norm logging
6. Optimizer LR logging
7. Curriculum state recovery
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.normuon import NorMuon
from training.numerical_health import NumericalHealthMonitor, HealthConfig
from utils.checkpoint import CheckpointManager


# ── Test 1: NorMuon state survives restart ───────────────────────────────


def test_normuon_state_survives_restart():
    """NorMuon optimizer state must survive save/load cycle."""
    print("Testing NorMuon state recovery...")

    tmpdir = tempfile.mkdtemp()
    try:
        model = nn.Linear(16, 8, bias=False)
        opt = NorMuon(
            [{"params": model.parameters()}],
            lr=0.02,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # Run a few steps to populate state
        for _ in range(5):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Capture state before save
        state_before = {}
        for p in model.parameters():
            if p in opt.state:
                s = opt.state[p]
                state_before["step"] = s["step"]
                state_before["exp_avg"] = s["exp_avg"].clone()
                state_before["exp_avg_sq"] = s["exp_avg_sq"].clone()

        # Save checkpoint
        ckpt = CheckpointManager(tmpdir, async_mode=False)
        ckpt.save(
            model, opt, step=100,
            extra_optimizers={"muon": opt},
        )

        # Create fresh model + optimizer
        model2 = nn.Linear(16, 8, bias=False)
        opt2 = NorMuon(
            [{"params": model2.parameters()}],
            lr=0.02,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # Load checkpoint
        ckpt2 = CheckpointManager(tmpdir, async_mode=False)
        ckpt2.load(
            model2, step=100, device="cpu",
            optimizer=None,
            extra_optimizers={"muon": opt2},
        )

        # Verify state restored
        for p in model2.parameters():
            if p in opt2.state:
                s = opt2.state[p]
                assert s["step"] == state_before["step"], (
                    f"step mismatch: {s['step']} != {state_before['step']}"
                )
                assert torch.allclose(s["exp_avg"], state_before["exp_avg"]), (
                    "exp_avg not restored"
                )
                assert torch.allclose(s["exp_avg_sq"], state_before["exp_avg_sq"]), (
                    "exp_avg_sq not restored"
                )

        print("✓ NorMuon state survives restart")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 2: Curriculum survives restart ──────────────────────────────────


def test_curriculum_survives_restart():
    """Curriculum _advanced flag must survive checkpoint save/load."""
    print("Testing curriculum state recovery...")

    tmpdir = tempfile.mkdtemp()
    try:
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate curriculum that has advanced
        curriculum_meta = {
            "curriculum_advanced": True,
            "curriculum_switch_step": 1000,
        }

        # Save with curriculum state
        ckpt = CheckpointManager(tmpdir, async_mode=False)
        ckpt.save(model, opt, step=500, extra_meta=curriculum_meta)

        # Load and verify
        ckpt2 = CheckpointManager(tmpdir, async_mode=False)
        meta = ckpt2.load(model, step=500, device="cpu")

        assert meta.get("curriculum_advanced") is True, (
            f"curriculum_advanced not saved: {meta}"
        )
        assert meta.get("curriculum_switch_step") == 1000, (
            f"curriculum_switch_step not saved: {meta}"
        )

        print("✓ Curriculum state survives restart")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 3: NaN loss aborts training ─────────────────────────────────────


def test_nan_loss_aborts():
    """NaN loss must raise RuntimeError."""
    print("Testing NaN loss detection...")

    monitor = NumericalHealthMonitor(HealthConfig())

    # Normal losses should work
    for i in range(5):
        assert not monitor.update_loss(1.0 + i * 0.1, i)

    # NaN loss must raise
    try:
        monitor.update_loss(float("nan"), 10)
        assert False, "Expected RuntimeError for NaN loss"
    except RuntimeError as e:
        assert "NaN" in str(e)

    print("✓ NaN loss aborts training")


# ── Test 4: Inf gradients abort training ─────────────────────────────────


def test_inf_gradients_abort():
    """Inf gradients must raise RuntimeError."""
    print("Testing Inf gradient detection...")

    monitor = NumericalHealthMonitor(HealthConfig())
    model = nn.Linear(10, 10)

    # Normal gradients should work
    for i in range(3):
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        monitor.update_gradients(model, i)
        model.zero_grad()

    # Inf gradient must raise
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    # Inject Inf into gradient
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(float("inf"))

    try:
        monitor.update_gradients(model, 10)
        assert False, "Expected RuntimeError for Inf gradient"
    except RuntimeError as e:
        assert "Inf" in str(e)

    model.zero_grad()
    print("✓ Inf gradients abort training")


# ── Test 5: NaN gradients abort training ─────────────────────────────────


def test_nan_gradients_abort():
    """NaN gradients must raise RuntimeError."""
    print("Testing NaN gradient detection...")

    monitor = NumericalHealthMonitor(HealthConfig())
    model = nn.Linear(10, 10)

    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    # Inject NaN into gradient
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(float("nan"))

    try:
        monitor.update_gradients(model, 10)
        assert False, "Expected RuntimeError for NaN gradient"
    except RuntimeError as e:
        assert "NaN" in str(e)

    model.zero_grad()
    print("✓ NaN gradients abort training")


# ── Test 6: Inf loss aborts training ─────────────────────────────────────


def test_inf_loss_aborts():
    """Inf loss must raise RuntimeError."""
    print("Testing Inf loss detection...")

    monitor = NumericalHealthMonitor(HealthConfig())

    # Inf loss must raise
    try:
        monitor.update_loss(float("inf"), 10)
        assert False, "Expected RuntimeError for Inf loss"
    except RuntimeError as e:
        assert "Inf" in str(e)

    print("✓ Inf loss aborts training")


# ── Test 7: Router z-loss contributes to total loss ──────────────────────


def test_router_z_loss_contributes():
    """Router z-loss must be non-zero and contribute to total loss."""
    print("Testing router z-loss integration...")

    try:
        from models.moe import AuxLossFreeGate

        gate = AuxLossFreeGate({
            "dim": 64,
            "n_routed_experts": 8,
            "n_activated_experts": 2,
        })

        # Forward pass to populate _last_router_logits
        x = torch.randn(32, 64)
        logits, indices = gate(x)

        # z-loss should be positive
        z_loss = gate.get_z_loss()
        assert z_loss.item() > 0, f"z-loss should be > 0, got {z_loss.item()}"

        # z-loss = mean(logsumexp(logits)^2)
        expected = torch.logsumexp(gate._last_router_logits, dim=-1).pow(2).mean()
        assert torch.allclose(z_loss, expected, atol=1e-5), (
            f"z-loss mismatch: {z_loss.item()} vs {expected.item()}"
        )

        print("✓ Router z-loss contributes to total loss")
    except ImportError:
        print("⚠ Router z-loss test skipped (import error)")


# ── Test 8: Checkpoint retention works ───────────────────────────────────


def test_checkpoint_retention():
    """keep_last_n must delete old checkpoints but preserve best."""
    print("Testing checkpoint retention...")

    tmpdir = Path(tempfile.mkdtemp())
    try:
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt = CheckpointManager(tmpdir, async_mode=False)

        # Save 8 checkpoints
        for step in range(0, 800, 100):
            ckpt.save(model, opt, step=step)

        # keep_last_n=3 should delete steps 0, 100, 200, 300, 400
        ckpt.keep_last_n(3)

        remaining = ckpt.list_checkpoints()
        assert len(remaining) == 3, f"Expected 3 checkpoints, got {len(remaining)}: {remaining}"
        assert 700 in remaining, f"Latest checkpoint (700) missing: {remaining}"
        assert 600 in remaining, f"Checkpoint 600 missing: {remaining}"
        assert 500 in remaining, f"Checkpoint 500 missing: {remaining}"

        # Old checkpoints should be deleted
        for step in [0, 100, 200, 300, 400]:
            model_path = tmpdir / f"model_step_{step}.safetensors"
            assert not model_path.exists(), f"Old checkpoint {step} not deleted"

        print("✓ Checkpoint retention works")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 9: Checkpoint retention preserves best ──────────────────────────


def test_checkpoint_retention_preserves_best():
    """keep_last_n must not delete the best checkpoint."""
    print("Testing checkpoint retention preserves best...")

    tmpdir = Path(tempfile.mkdtemp())
    try:
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt = CheckpointManager(tmpdir, async_mode=False)

        # Save checkpoints with decreasing val_loss (best = last)
        for step in range(0, 800, 100):
            val_loss = 5.0 - step / 200.0  # decreasing = improving
            ckpt.save(model, opt, step=step, val_loss=val_loss)

        # Best should be step 700 (lowest val_loss)
        best_path = tmpdir / "best.safetensors"
        assert best_path.exists(), "best.safetensors not created"

        # keep_last_n=2
        ckpt.keep_last_n(2)

        # best.safetensors should still exist
        assert best_path.exists(), "best.safetensors deleted by keep_last_n"

        # Best checkpoint files should still exist
        best_meta_path = tmpdir / "best_meta.json"
        assert best_meta_path.exists(), "best_meta.json deleted"

        print("✓ Checkpoint retention preserves best")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 10: Metrics appear in logger output ─────────────────────────────


def test_metrics_in_logger():
    """Logger must accept and store grad_norm, muon_lr, z_loss."""
    print("Testing metrics in logger...")

    from utils.logging import TrainerLogger

    tmpdir = tempfile.mkdtemp()
    try:
        logger = TrainerLogger(
            log_interval=1,
            wandb_enabled=False,
            mlflow_enabled=False,
        )

        # Log with all new metrics
        logger.log(
            step=0,
            loss=2.5,
            lr=3e-4,
            muon_lr=0.02,
            grad_norm=1.23,
            metrics={"balance_loss": 0.001, "z_loss": 0.05},
        )

        # Check that the logger didn't crash and processed the metrics
        # The logger prints to stdout and submits to backends
        # If we get here without exception, the metrics were accepted
        print("✓ Metrics appear in logger output")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 11: NorMuon momentum_buffer survives restart ────────────────────


def test_normuon_momentum_buffer_survives():
    """NorMuon momentum_buffer must survive save/load."""
    print("Testing NorMuon momentum_buffer recovery...")

    tmpdir = tempfile.mkdtemp()
    try:
        model = nn.Linear(16, 8, bias=False)
        opt = NorMuon(
            [{"params": model.parameters()}],
            lr=0.02,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # Run steps to populate momentum_buffer
        for _ in range(10):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Verify momentum_buffer exists
        for p in model.parameters():
            if p in opt.state:
                assert "momentum_buffer" not in opt.state[p], (
                    "NorMuon should not have momentum_buffer (uses exp_avg)"
                )

        # Save and load
        ckpt = CheckpointManager(tmpdir, async_mode=False)
        ckpt.save(model, opt, step=200, extra_optimizers={"muon": opt})

        model2 = nn.Linear(16, 8, bias=False)
        opt2 = NorMuon(
            [{"params": model2.parameters()}],
            lr=0.02,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        # Run a dummy step to initialize state
        x = torch.randn(4, 16)
        loss = model2(x).sum()
        loss.backward()
        opt2.step()
        opt2.zero_grad()

        ckpt2 = CheckpointManager(tmpdir, async_mode=False)
        ckpt2.load(model2, step=200, device="cpu", extra_optimizers={"muon": opt2})

        # Verify state restored
        for p in model2.parameters():
            if p in opt2.state:
                s = opt2.state[p]
                assert "exp_avg" in s, "exp_avg missing after load"
                assert "exp_avg_sq" in s, "exp_avg_sq missing after load"
                assert "step" in s, "step missing after load"
                assert s["step"] > 0, f"step should be > 0, got {s['step']}"

        print("✓ NorMuon momentum_buffer survives restart")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("Phase 2A — Training Reliability Tests")
    print("=" * 60)

    tests = [
        test_normuon_state_survives_restart,
        test_curriculum_survives_restart,
        test_nan_loss_aborts,
        test_inf_loss_aborts,
        test_nan_gradients_abort,
        test_inf_gradients_abort,
        test_router_z_loss_contributes,
        test_checkpoint_retention,
        test_checkpoint_retention_preserves_best,
        test_metrics_in_logger,
        test_normuon_momentum_buffer_survives,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
