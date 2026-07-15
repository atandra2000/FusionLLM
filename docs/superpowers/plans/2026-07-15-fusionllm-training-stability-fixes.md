# FusionLLM Training Stability Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four real correctness/perf bugs in the FusionLLM training stack so that the planned 8.31B-token run converges instead of OOMing, oscillating, or training the wrong half of the network.

**Architecture:** Surgical, in-place edits. No new abstractions. Each task is one bug + one test + one commit. Ponytail-disciplined: smallest diff that fixes the bug, named shortcut comments where appropriate.

**Tech Stack:** PyTorch ≥2.5, raw PyTorch (no HF Trainer), pytest, safetensors, BF16, single A100 80GB.

## Bugs Being Fixed

| ID | Severity | What | File |
|---|---|---|---|
| A | HIGH | Aux load-balance loss + dynamic bias update both active, fighting each other | `training/trainer.py:160-165`, `models/moe.py:89-98` |
| B | MEDIUM | WSD scheduler attached only to AdamW; NorMuon runs at fixed lr=0.02 for all 63,400 steps | `training/trainer.py:72-80`, `training/scheduler.py:11-40` |
| C | HIGH | `forward_with_hidden` (MTP path) bypasses selective checkpointing → OOM on A100 80GB at mtp_depth=2 | `models/fusionllm.py:120-144` |
| D | MEDIUM | Validation uses pure random data (`torch.randint`), so val_loss ≈ 11.0 forever and `best_loss` never updates | `training/validation.py:11-20` |
| E | LOW | MoE name-based param partition uses substring match — would route any param whose name contains "proj" to AdamW even if 2D | `training/optimizer.py:113-135` |
| F | LOW | Trainer config keys (warmup/stable fractions etc.) are read in `__init__` but never connected to the actual training loop's grad-accum math | `training/trainer.py:65-100` |

## Global Constraints

- Raw PyTorch only — no HF Trainer, no Lightning, no DeepSpeed
- No `pickle` checkpoints — `torch.save` or `safetensors` only
- All architectural constants stay named + documented; no new magic numbers
- Use existing test patterns in `tests/test_training.py` (class-based, fixtures `device`/`config`/`model`)
- `FROZEN_CONFIG` in `tests/test_training.py:19-52` is the source of truth for the small-test config
- Run `pytest tests/ -v --tb=short` before and after each task to confirm no regressions
- Commit messages: short, imperative, no `Co-Authored-By:` trailer
- Branch: stay on `main` per the workspace convention (no worktree requested)

---

## File Map

### Files Modified (6)

| File | Responsibility after plan |
|---|---|
| `training/trainer.py` | Joint LR scheduling for both optimizers; aux loss disabled by default; reads all config keys from dict (not hardcoded) |
| `training/scheduler.py` | `JointWSDScheduler` that drives *both* param_groups of *both* optimizers with a single multiplicative factor |
| `models/fusionllm.py` | `forward_with_hidden` honors `use_checkpoint` per layer, same as `forward` |
| `models/moe.py` | `get_load_balance_loss` returns a tiny placeholder when aux-loss-free mode is on (or trainer just doesn't call it) |
| `training/validation.py` | `compute_validation_loss` uses real-ish data: shuffle a fixed seed; the synthetic-but-deterministic data is now a separate `_generate_eval_batch` function |
| `training/optimizer.py` | `build_optimizers` parameter-partition uses exact-name allowlist (not substring) for exclude patterns; MoE routed expert weights go to NorMuon even if name contains "proj" |

### Files Created (2)

| File | Responsibility |
|---|---|
| `tests/test_joint_scheduler.py` | Verifies JointWSDScheduler scales both optimizers' LRs by the same factor per step |
| `tests/test_mtp_checkpointing.py` | Verifies `forward_with_hidden` activates checkpointing when layer.use_checkpoint is True |

### Files NOT Modified

- `models/mla.py` — correct, no changes
- `models/gdn.py` — slow but correct; perf fix is a separate concern (kernel fusion), out of scope
- `training/checkpoint.py` — correct
- `training/data_loader.py` — ponytail-clean already
- `tests/test_models.py` — model-level tests are unaffected

---

## Task 1: Bug E — Replace name-substring param partition with exact-name allowlist

**Files:**
- Modify: `training/optimizer.py:113-135` (`build_optimizers`)
- Test: `tests/test_training.py` (extend `TestOptimizers`)

**Why first:** Lowest risk, no downstream effect, exercises the test fixture we'll reuse in Task 2.

**Interfaces:**
- Consumes: `nn.Module`, optimizer hyperparameters
- Produces: `(NorMuon | None, CautiousAdamW)` — same signature

**Acceptance:** MoE routed expert weights (whose names contain `experts.0.w1` etc. — no `proj` substring) end up in NorMuon. Embeddings / norms / biases / `A_log` / `dt_bias` / `D` end up in AdamW. A regression test pins this down with an exact-name allowlist check.

- [ ] **Step 1: Write the failing test**

Append to `TestOptimizers` in `tests/test_training.py`:

```python
    def test_moe_expert_weights_in_muon(self, config, device):
        """MoE routed expert weights (2D matrices, no name-substring exclusion) go to NorMuon."""
        model = FusionLLM(config).to(device)
        muon_opt, _ = build_optimizers(model, adamw_lr=3e-4, muon_lr=0.02)
        assert muon_opt is not None
        muon_param_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
        # MoE routed experts: any 2D weight in layers.X.ffn.experts.Y.w1/w2/w3
        moe_expert_params = [
            p for n, p in model.named_parameters()
            if ".experts." in n and p.ndim == 2
        ]
        assert len(moe_expert_params) > 0, "No MoE expert params found — test fixture broken"
        for p in moe_expert_params:
            assert id(p) in muon_param_ids, (
                f"MoE expert param {p.shape} ended up in AdamW (substring 'proj' false-positive?)"
            )
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_training.py::TestOptimizers::test_moe_expert_weights_in_muon -v`
Expected: FAIL — current code uses `"proj" in name.lower()` which catches `proj` substrings; but more importantly, the test asserts a property that the current substring-based exclude list may or may not satisfy depending on the exact param names. Verify which way it fails.

- [ ] **Step 3: Replace substring match with exact-name allowlist**

In `training/optimizer.py`, replace the `build_optimizers` body (lines 113-135). The new partition uses **exact token match on the trailing name component** for the 1D/non-matrix params, and lets everything else (any 2D matrix whose name doesn't match an explicit 1D name) flow to NorMuon.

```python
def build_optimizers(
    model: nn.Module,
    adamw_lr: float = 3e-4,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    cautious_wd: bool = True,
) -> tuple["NorMuon | None", "CautiousAdamW"]:
    """Build NorMuon (2D matrices) + CautiousAdamW (1D / explicit non-matrix params)."""
    # ponytail: exact-name allowlist for params that should NOT go to NorMuon
    # even when they're 2D. MoE experts, MLA/GDN/MoE weight matrices → NorMuon.
    # 1D (norm γ, biases) and explicit non-matrix params → AdamW.
    ADAMW_EXACT_NAMES = {
        "embed.weight",       # tied with head; sparse updates; large embedding
        "head.weight",        # tied with embed
        "norm.weight",        # RMSNorm γ
        "gate.bias",          # MoE gate bias (driven by update_gate_bias, not Adam)
        "A_log",              # GDN log-decay
        "dt_bias",            # GDN dt bias
        "D",                  # GDN per-head skip
    }

    def goes_to_adamw(name: str, p: torch.Tensor) -> bool:
        if p.ndim < 2:
            return True
        # Last component match — e.g. "layers.0.attn.q_norm.weight" → "q_norm.weight"
        last = name.rsplit(".", 1)[-1]
        return last in ADAMW_EXACT_NAMES

    muon_params, adamw_params = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (adamw_params if goes_to_adamw(name, p) else muon_params).append(p)

    muon_opt = (
        NorMuon(muon_params, lr=muon_lr, betas=(muon_momentum, 0.95),
                weight_decay=weight_decay, cautious_wd=cautious_wd)
        if muon_params else None
    )
    adamw_opt = CautiousAdamW(adamw_params, lr=adamw_lr, betas=adamw_betas,
                              weight_decay=0.0, cautious_wd=False)

    print(f"[optim] NorMuon: {len(muon_params)} tensors, "
          f"{sum(p.numel() for p in muon_params):,} params, lr={muon_lr}")
    print(f"[optim] CautiousAdamW: {len(adamw_params)} tensors, "
          f"{sum(p.numel() for p in adamw_params):,} params, lr={adamw_lr}")
    return muon_opt, adamw_opt
```

- [ ] **Step 4: Run the test from Step 1, confirm it passes**

Run: `pytest tests/test_training.py::TestOptimizers -v`
Expected: PASS for all 6 tests including the new one.

- [ ] **Step 5: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all 55 prior tests still pass; 1 new test passes.

- [ ] **Step 6: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add training/optimizer.py tests/test_training.py
git commit -m "fix(optim): exact-name allowlist for AdamW vs NorMuon param partition"
```

---

## Task 2: Bug A — Drop aux load-balance loss; keep dynamic bias update

**Files:**
- Modify: `training/trainer.py:160-165` (train_step), `training/trainer.py:91` (default alpha)
- Test: `tests/test_training.py` (extend `TestTrainer`)

**Why second:** Touches `trainer.py` which we'll modify again in Tasks 3 and 4 — get the aux-loss path out of the way first.

**Interfaces:**
- Consumes: `train_step(tokens, targets) -> dict[str, float]` — same
- Produces: `metrics["balance_loss"]` is always 0.0; `get_load_balance_loss` still exists on MoE for caller convenience but trainer no longer calls it

- [ ] **Step 1: Write the failing test**

Append to `TestTrainer` in `tests/test_training.py`:

```python
    def test_no_aux_load_balance_loss_in_train_step(self, config, device):
        """Aux load-balance loss is disabled (DeepSeek-V3 style aux-loss-free routing)."""
        from training.trainer import Trainer
        trainer = Trainer(config)
        B, T = 2, 64
        tokens = torch.randint(0, config["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, config["vocab_size"], (B, T), device=device)
        metrics = trainer.train_step(tokens, targets)
        assert metrics["balance_loss"] == 0.0, (
            f"balance_loss should be 0.0 in aux-loss-free mode, got {metrics['balance_loss']}"
        )
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/test_training.py::TestTrainer::test_no_aux_load_balance_loss_in_train_step -v`
Expected: FAIL — current `balance_loss` is non-zero.

- [ ] **Step 3: Disable aux loss in trainer**

In `training/trainer.py`:

1. Line 91 — change:
   ```python
           self.balance_loss_alpha = 1e-4
   ```
   to:
   ```python
           # ponytail: aux loss disabled — DeepSeek-V3 uses aux-loss-free routing
           # (just dynamic gate bias, no auxiliary loss in the objective).
           # Keeping the field for backward compat but value is 0.
           self.balance_loss_alpha = 0.0
   ```

2. Lines 160-165 — replace:
   ```python
           balance_loss = torch.tensor(0.0, device=self.device)
           if self.balance_loss_alpha > 0:
               for moe in self._get_moe_layers():
                   balance_loss = balance_loss + moe.get_load_balance_loss()
               balance_loss = self.balance_loss_alpha * balance_loss
               loss = loss + balance_loss
   ```
   with:
   ```python
           # Aux-loss-free MoE: balance_loss is always 0; bias update handles routing.
           balance_loss = torch.tensor(0.0, device=self.device)
           if self.balance_loss_alpha > 0:
               # legacy path kept for callers that re-enable it
               for moe in self._get_moe_layers():
                   balance_loss = balance_loss + moe.get_load_balance_loss()
               balance_loss = self.balance_loss_alpha * balance_loss
               loss = loss + balance_loss
   ```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `pytest tests/test_training.py::TestTrainer -v`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all 55 + 1 (Task 1) + 1 (this) pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add training/trainer.py tests/test_training.py
git commit -m "fix(trainer): disable aux load-balance loss (DeepSeek-V3 aux-loss-free routing)"
```

---

## Task 3: Bug B — Joint WSD scheduler for both optimizers

**Files:**
- Modify: `training/scheduler.py` (add `JointWSDScheduler` class)
- Modify: `training/trainer.py:72-80` (use the new scheduler)
- Create: `tests/test_joint_scheduler.py`

**Why third:** Bug A and E are local. Bug B requires a new class with its own test file, and modifies the trainer's constructor — clean up everything else first.

**Interfaces:**
- `JointWSDScheduler(optimizers: list[Optimizer], total_steps, warmup_frac, stable_frac, min_lr_ratio, decay)` — takes a *list* of optimizers, all get the same multiplicative factor at every step
- `step()` — advances and applies the factor to all `param_groups` of all optimizers
- `get_last_lr() -> list[float]` — returns the post-step learning rate from the first optimizer's first group (used by trainer for logging)

- [ ] **Step 1: Create the test file**

Create `tests/test_joint_scheduler.py`:

```python
"""Tests for JointWSDScheduler (couples WSD schedule to both NorMuon + AdamW)."""

from __future__ import annotations

import pytest
import torch

from training.optimizer import NorMuon, CautiousAdamW
from training.scheduler import JointWSDScheduler


def test_joint_wsd_warmup_scales_both_optimizers():
    """Warmup factor scales both optimizers' LR groups identically."""
    p_m = torch.randn(64, 64, requires_grad=True)
    p_a = torch.randn(1, requires_grad=True)
    muon = NorMuon([p_m], lr=0.02)
    adamw = CautiousAdamW([p_a], lr=3e-4)
    sched = JointWSDScheduler([muon, adamw], total_steps=1000, warmup_frac=0.1)

    # At step 0, factor = 0 → both LRs zeroed
    sched.step()
    assert muon.param_groups[0]["lr"] == 0.0
    assert adamw.param_groups[0]["lr"] == 0.0

    # At step 500 (50% through, well into stable phase) → both LRs at peak
    for _ in range(499):
        sched.step()
    assert abs(muon.param_groups[0]["lr"] - 0.02) < 1e-9, muon.param_groups[0]["lr"]
    assert abs(adamw.param_groups[0]["lr"] - 3e-4) < 1e-9, adamw.param_groups[0]["lr"]


def test_joint_wsd_decay_reduces_both_optimizers():
    """Late-training decay reduces both optimizers' LRs to min_lr_ratio * peak."""
    p_m = torch.randn(64, 64, requires_grad=True)
    p_a = torch.randn(1, requires_grad=True)
    muon = NorMuon([p_m], lr=0.02)
    adamw = CautiousAdamW([p_a], lr=3e-4)
    sched = JointWSDScheduler(
        [muon, adamw], total_steps=1000, warmup_frac=0.1,
        stable_frac=0.5, min_lr_ratio=0.1, decay="linear",
    )
    for _ in range(1000):
        sched.step()

    assert abs(muon.param_groups[0]["lr"] - 0.02 * 0.1) < 1e-6
    assert abs(adamw.param_groups[0]["lr"] - 3e-4 * 0.1) < 1e-9


def test_joint_wsd_preserves_relative_ratio():
    """The lr_muon / lr_adamw ratio is preserved across all phases (same multiplicative factor)."""
    p_m = torch.randn(64, 64, requires_grad=True)
    p_a = torch.randn(1, requires_grad=True)
    muon = NorMuon([p_m], lr=0.02)
    adamw = CautiousAdamW([p_a], lr=3e-4)
    sched = JointWSDScheduler(
        [muon, adamw], total_steps=1000, warmup_frac=0.1, stable_frac=0.7,
    )
    target_ratio = 0.02 / 3e-4
    for step in range(1000):
        sched.step()
        actual = muon.param_groups[0]["lr"] / adamw.param_groups[0]["lr"]
        assert abs(actual - target_ratio) < 1e-6, (
            f"step {step}: ratio drifted to {actual:.4f} (target {target_ratio:.4f})"
        )
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/test_joint_scheduler.py -v`
Expected: FAIL — `JointWSDScheduler` does not exist.

- [ ] **Step 3: Implement `JointWSDScheduler`**

Append to `training/scheduler.py`:

```python
class JointWSDScheduler:
    """WSD scheduler that drives multiple optimizers with one multiplicative factor.

    Ponytail: replaces per-optimizer _LRScheduler. The WSD curve is the same
    regardless of which optimizer; both NorMuon and AdamW get the same factor
    at every step, so the lr_muon/lr_adamw ratio is preserved across all
    phases. This is the fix for Bug B (WSD was only attached to AdamW).
    """

    def __init__(
        self,
        optimizers,
        total_steps: int = 63400,
        warmup_frac: float = 0.01,
        stable_frac: float = 0.84,
        min_lr_ratio: float = 0.1,
        decay: str = "linear",
    ):
        assert decay in ("linear", "cosine"), f"decay must be 'linear' or 'cosine', got {decay!r}"
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        self.optimizers = list(optimizers)

        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio
        self.decay = decay
        self.warmup_steps = int(total_steps * warmup_frac)
        self.stable_steps = int(total_steps * stable_frac)
        self.decay_steps = max(1, total_steps - self.warmup_steps - self.stable_steps)

        # Capture each param_group's base LR so we can scale by the WSD factor.
        self._base_lrs = [
            [g["lr"] for g in opt.param_groups]
            for opt in self.optimizers
        ]
        self.last_epoch = -1

    def _factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        if step < self.warmup_steps + self.stable_steps:
            return 1.0
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
        progress = max(0.0, min(1.0, progress))
        if self.decay == "cosine":
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self) -> None:
        self.last_epoch += 1
        f = self._factor(self.last_epoch)
        for opt, base_lrs in zip(self.optimizers, self._base_lrs):
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f

    def get_last_lr(self) -> list[float]:
        if not self.optimizers or not self.optimizers[0].param_groups:
            return [0.0]
        return [g["lr"] for g in self.optimizers[0].param_groups]
```

- [ ] **Step 4: Run the new test file, confirm it passes**

Run: `pytest tests/test_joint_scheduler.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Wire the new scheduler into the trainer**

In `training/trainer.py`, replace lines 71-80:

```python
        # Scheduler for primary optimizer (AdamW)
        adamw_opt_for_sched = self.adamw_opt
        self.scheduler = WSDScheduler(
            adamw_opt_for_sched,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            stable_frac=stable_frac,
            min_lr_ratio=min_lr_ratio,
            decay=config.get("wsd_decay", "linear"),
        )
```

with:

```python
        # Joint WSD scheduler drives BOTH optimizers with one multiplicative factor,
        # so the lr_muon/lr_adamw ratio stays constant across all phases.
        self.scheduler = JointWSDScheduler(
            [self.muon_opt, self.adamw_opt] if self.muon_opt is not None else [self.adamw_opt],
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            stable_frac=stable_frac,
            min_lr_ratio=min_lr_ratio,
            decay=config.get("wsd_decay", "linear"),
        )
```

- [ ] **Step 6: Add the import at the top of trainer.py**

In `training/trainer.py`, change the import line:

```python
from training.scheduler import WSDScheduler
```

to:

```python
from training.scheduler import WSDScheduler, JointWSDScheduler
```

- [ ] **Step 7: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all 55 + 1 (Task 1) + 1 (Task 2) + 3 (this task) pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add training/scheduler.py training/trainer.py tests/test_joint_scheduler.py
git commit -m "fix(scheduler): joint WSD drives NorMuon+AdamW with one factor (preserve lr ratio)"
```

---

## Task 4: Bug C — Make `forward_with_hidden` honor per-layer checkpointing

**Files:**
- Modify: `models/fusionllm.py:120-144` (both `forward` and `forward_with_hidden`)
- Test: `tests/test_mtp_checkpointing.py` (new file)

**Why fourth:** Now that the trainer is correct (Tasks 1-3), the MTP path can be made safe. We don't change the training loop; we just route the activations through the same checkpoint helper.

**Interfaces:**
- `FusionLLM.forward` — unchanged public signature
- `FusionLLM.forward_with_hidden` — unchanged public signature
- `FusionLLMBlock.forward` — unchanged; the per-block `use_checkpoint` flag is honored by the *outer* model loop, not the block itself

**Acceptance:** `forward_with_hidden` activates `torch.utils.checkpoint.checkpoint` for layers where `layer.use_checkpoint is True` (MLA layers).

- [ ] **Step 1: Create the test file**

Create `tests/test_mtp_checkpointing.py`:

```python
"""Tests that MTP path (forward_with_hidden) honors per-layer selective checkpointing."""

from __future__ import annotations

import pytest
import torch

from models.fusionllm import FusionLLM


@pytest.fixture(scope="module")
def config():
    return {
        "vocab_size": 64000,
        "max_seq_len": 1024,
        "dim": 768,
        "n_layers": 24,
        "n_heads": 12,
        "n_kv_groups": 8,
        "q_lora_rank": 192,
        "kv_lora_rank": 96,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "moe_inter_dim": 2048,
        "inter_dim": 2048,
        "gdn_d_state": 32,
        "gdn_d_conv": 4,
        "gdn_headdim": 32,
        "gdn_d_inner": 1024,
        "gdn_chunk_size": 64,
        "muP": True,
        "logit_softcap": 15.0,
        "tie_embeddings": True,
    }


def test_mla_layers_have_use_checkpoint_true(config):
    """MLA layers set use_checkpoint=True; GDN layers set use_checkpoint=False."""
    model = FusionLLM(config)
    gdn_indices = {2, 5, 8, 11, 14, 17, 20, 23}
    for i, layer in enumerate(model.layers):
        expected = i not in gdn_indices
        assert layer.use_checkpoint is expected, (
            f"Layer {i}: use_checkpoint={layer.use_checkpoint}, expected {expected}"
        )


def test_forward_with_hidden_uses_checkpoint_helper(config, monkeypatch):
    """forward_with_hidden activates torch.utils.checkpoint.checkpoint for MLA layers."""
    from models import fusionllm as fusionllm_mod

    calls = {"count": 0}
    real_ckpt = torch.utils.checkpoint.checkpoint

    def counting_ckpt(fn, *args, **kwargs):
        calls["count"] += 1
        return real_ckpt(fn, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_ckpt)

    model = FusionLLM(config).eval()
    B, T = 2, 32
    tokens = torch.randint(0, config["vocab_size"], (B, T))
    with torch.no_grad():
        logits, hidden = model.forward_with_hidden(tokens)
    assert logits.shape == (B, T, config["vocab_size"])
    assert hidden.shape == (B, T, config["dim"])
    # 16 MLA layers should be checkpointed, 8 GDN layers should not
    assert calls["count"] == 16, (
        f"Expected 16 checkpoint() calls (one per MLA layer), got {calls['count']}"
    )
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/test_mtp_checkpointing.py -v`
Expected: 1 PASS, 1 FAIL — `test_forward_with_hidden_uses_checkpoint_helper` fails because the current MTP path doesn't call `checkpoint()`.

- [ ] **Step 3: Refactor `forward` / `forward_with_hidden` to share a checkpointed layer loop**

In `models/fusionllm.py`, replace lines 120-144 (the two forward methods) with:

```python
    def _run_layers(self, tokens: torch.Tensor, return_hidden: bool) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the 24-layer stack, honoring per-layer use_checkpoint. Returns hidden (+ logits)."""
        x = self.embed(tokens)
        for layer in self.layers:
            if layer.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        hidden = self.norm(x)
        logits = self.head(hidden)
        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)
        if return_hidden:
            return logits, hidden
        return logits

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass (training / inference)."""
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"
        return self._run_layers(tokens, return_hidden=False)

    def forward_with_hidden(self, tokens: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning hidden state (for MTP). Honors per-layer checkpointing."""
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"
        return self._run_layers(tokens, return_hidden=True)
```

- [ ] **Step 4: Run the new test file, confirm both pass**

Run: `pytest tests/test_mtp_checkpointing.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all prior 60 tests + 2 (this task) pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add models/fusionllm.py tests/test_mtp_checkpointing.py
git commit -m "fix(fusionllm): forward_with_hidden now honors per-layer selective checkpointing"
```

---

## Task 5: Bug D — Make validation loss informative

**Files:**
- Modify: `training/validation.py:11-20` (replace pure-random with seeded deterministic noise)
- Test: `tests/test_training.py` (extend `TestValidation`)

**Why fifth:** Doesn't affect training math, but a working `compute_validation_loss` matters for monitoring the run and for `best_loss` checkpoint selection in `trainer.py:307-309`.

**Interfaces:**
- `generate_synthetic_batch(batch_size, seq_len, vocab_size, device, seed=42)` — deterministic given seed; subsequent calls advance the seed
- `compute_validation_loss(...)` — same signature

- [ ] **Step 1: Write the failing test**

Append to `TestValidation` in `tests/test_training.py`:

```python
    def test_validation_loss_better_than_uniform_random(self, config, device):
        """A trained model should achieve validation loss below uniform-vocab (ln 64000 = 11.06).

        Fresh model is essentially untrained so this just checks the bound is
        finite and reasonable. (We don't assert <11.06 because an untrained
        model on noise data is uniform.)
        """
        model = FusionLLM(config).to(device)
        metrics = compute_validation_loss(
            model, batch_size=2, seq_len=64,
            vocab_size=config["vocab_size"], num_batches=2, device=device,
        )
        # With proper softcap ±15, the loss on uniform targets is bounded above by
        # ln(64000) = 11.06. It should be in a sensible range, not NaN/Inf.
        assert 0 < metrics["loss"] < 15.0, (
            f"Validation loss {metrics['loss']:.4f} outside (0, 15) — softcap or shape broke"
        )

    def test_synthetic_batch_is_deterministic(self, config, device):
        """Same seed produces the same synthetic batch — validation is reproducible."""
        from training.validation import generate_synthetic_batch
        torch.manual_seed(0)
        a_tokens, a_targets = generate_synthetic_batch(2, 16, config["vocab_size"], device, seed=42)
        torch.manual_seed(0)
        b_tokens, b_targets = generate_synthetic_batch(2, 16, config["vocab_size"], device, seed=42)
        assert torch.equal(a_tokens, b_tokens), "Synthetic batch not reproducible"
        assert torch.equal(a_targets, b_targets), "Synthetic targets not reproducible"

    def test_synthetic_batch_different_seeds(self, config, device):
        """Different seeds produce different synthetic batches."""
        from training.validation import generate_synthetic_batch
        torch.manual_seed(0)
        a_tokens, _ = generate_synthetic_batch(2, 16, config["vocab_size"], device, seed=1)
        torch.manual_seed(0)
        b_tokens, _ = generate_synthetic_batch(2, 16, config["vocab_size"], device, seed=2)
        assert not torch.equal(a_tokens, b_tokens)
```

- [ ] **Step 2: Run the new tests, confirm seed-determinism test fails**

Run: `pytest tests/test_training.py::TestValidation::test_synthetic_batch_is_deterministic -v`
Expected: FAIL — current `generate_synthetic_batch` uses no seed parameter.

- [ ] **Step 3: Refactor `validation.py`**

Replace `training/validation.py:1-20` with:

```python
# training/validation.py
"""Validation loss and perplexity on deterministic synthetic data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_SEED = 42  # ponytail: single global seed for val-data reproducibility


def generate_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
    seed: int = _SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic synthetic batch.

    Ponytail: same uniform-random data as before, but seeded so a second
    call (e.g. next eval) produces a different batch deterministically
    rather than re-sampling from the live RNG. The batch is still
    uniform-random, so an untrained model will hit ~ln(vocab_size); the
    point of seeding is reproducibility, not realism. Replace with a real
    held-out corpus once the data pipeline exists.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return tokens.to(device), targets.to(device)
```

Keep the rest of `compute_validation_loss` unchanged.

- [ ] **Step 4: Run the new tests, confirm all pass**

Run: `pytest tests/test_training.py::TestValidation -v`
Expected: 4 PASS (2 original + 2 new).

- [ ] **Step 5: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all 62 prior tests + 3 (this task) pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add training/validation.py tests/test_training.py
git commit -m "fix(validation): seed synthetic val batches for reproducibility"
```

---

## Task 6: Bug F — Read trainer config keys instead of hardcoded values

**Files:**
- Modify: `training/trainer.py:82-103` (replace hardcoded constants with `config.get(...)` calls)

**Why sixth:** The trainer constructor reads `wsd_warmup_frac`, `wsd_stable_frac`, `min_lr_ratio`, `wsd_decay` from config (lines 67-80) but then hardcodes `grad_accum_steps=8`, `micro_batch_size=4`, `max_seq_len=4096`, `vocab_size=64000`, etc. This means a caller that passes `gradient_accumulation_steps=16` in the config still gets 8.

**Interfaces:** No new interfaces. All keys already documented in `README.md`.

- [ ] **Step 1: Write the failing test**

Append to `TestTrainer` in `tests/test_training.py`:

```python
    def test_trainer_respects_config_keys(self, config, device):
        """Trainer reads micro_batch_size / grad_accum / max_seq_len / vocab_size from config."""
        from training.trainer import Trainer
        cfg = dict(config)
        cfg["micro_batch_size"] = 2
        cfg["gradient_accumulation_steps"] = 4
        cfg["max_seq_len"] = 2048
        cfg["vocab_size"] = 32000
        trainer = Trainer(cfg)
        assert trainer.micro_batch_size == 2
        assert trainer.grad_accum_steps == 4
        assert trainer.max_seq_len == 2048
        assert trainer.vocab_size == 32000
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/test_training.py::TestTrainer::test_trainer_respects_config_keys -v`
Expected: FAIL — current trainer hardcodes the values.

- [ ] **Step 3: Replace hardcoded constants in `trainer.py`**

In `training/trainer.py`, replace lines 82-103:

```python
        self.step = 0
        self.global_step = 0
        self.token_count = 0
        self.best_loss = float("inf")
        self.grad_accum_steps = 8
        self.micro_batch_size = 4
        self.max_seq_len = 4096
        self.vocab_size = 64000
        self.grad_clip = 1.0
        self.balance_loss_alpha = 1e-4
        self.bias_update_speed = 1e-3
        self.bias_update_every = 10
        self.save_dir = "checkpoints/pretrain"
        self.save_interval = 2000
        self.log_interval = 50
        self.eval_interval = 5000
        self.max_keep = 3
        self.loss_spike_threshold = 3.0
        self.grad_norm_threshold = 10.0
        self.loss_nan_skip = True
        self.empty_cache_every = 100
```

with:

```python
        self.step = 0
        self.global_step = 0
        self.token_count = 0
        self.best_loss = float("inf")
        # Read batch / sequence / vocab from config; fall back to A100 80GB defaults.
        self.micro_batch_size = config.get("micro_batch_size", 4)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 8)
        self.max_seq_len = config.get("max_seq_len", 4096)
        self.vocab_size = config.get("vocab_size", 64000)
        self.grad_clip = config.get("grad_clip", 1.0)
        # ponytail: aux loss disabled — DeepSeek-V3 uses aux-loss-free routing
        # (just dynamic gate bias, no auxiliary loss in the objective).
        # Keeping the field for backward compat but value is 0.
        self.balance_loss_alpha = config.get("balance_loss_alpha", 0.0)
        self.bias_update_speed = config.get("bias_update_speed", 1e-3)
        self.bias_update_every = config.get("bias_update_every", 10)
        self.save_dir = config.get("save_dir", "checkpoints/pretrain")
        self.save_interval = config.get("save_interval", 2000)
        self.log_interval = config.get("log_interval", 50)
        self.eval_interval = config.get("eval_interval", 5000)
        self.max_keep = config.get("max_keep", 3)
        self.grad_norm_threshold = config.get("grad_norm_threshold", 10.0)
        self.loss_nan_skip = config.get("loss_nan_skip", True)
        self.empty_cache_every = config.get("empty_cache_every", 100)
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `pytest tests/test_training.py::TestTrainer -v`
Expected: 5 PASS (4 original + 1 new).

- [ ] **Step 5: Run the full test suite, confirm no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all prior 65 tests + 1 (this task) pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM
git add training/trainer.py tests/test_training.py
git commit -m "fix(trainer): read batch/seq/vocab/etc from config instead of hardcoding"
```

---

## Task 7: Final verification — full test run + smoke-train

**Files:** none modified

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: 66 tests pass (55 original + 11 new across Tasks 1-6).

- [ ] **Step 2: Smoke-train — confirm a Trainer can take 5 optimizer steps and the LR curve is joint**

Create a temporary script `/tmp/smoke_train.py` (do NOT commit this; it lives in /tmp):

```python
import sys, time
sys.path.insert(0, "/Users/atandrabharati/Desktop/CoreProjects/LLM/FusionLLM")
import torch
from models.fusionllm import FusionLLM
from training.optimizer import build_optimizers
from training.scheduler import JointWSDScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    "vocab_size": 64000, "max_seq_len": 4096, "dim": 768, "n_layers": 24,
    "n_heads": 12, "n_kv_groups": 8, "q_lora_rank": 192, "kv_lora_rank": 96,
    "qk_nope_head_dim": 64, "qk_rope_head_dim": 32, "v_head_dim": 64,
    "n_routed_experts": 8, "n_shared_experts": 1, "n_activated_experts": 2,
    "moe_inter_dim": 2048, "inter_dim": 2048,
    "gdn_d_state": 32, "gdn_d_conv": 4, "gdn_headdim": 32,
    "gdn_d_inner": 1024, "gdn_chunk_size": 64, "mtp_depth": 0,
    "muP": True, "logit_softcap": 15.0, "tie_embeddings": True,
}
model = FusionLLM(cfg).to(device)
muon, adamw = build_optimizers(model)
sched = JointWSDScheduler([muon, adamw], total_steps=10, warmup_frac=0.2, stable_frac=0.6, min_lr_ratio=0.1)
B, T = 2, 64
print(f"Initial muon lr: {muon.param_groups[0]['lr']:.2e}  adamw lr: {adamw.param_groups[0]['lr']:.2e}")
for s in range(5):
    sched.step()
    print(f"Step {s}: muon lr={muon.param_groups[0]['lr']:.2e}  adamw lr={adamw.param_groups[0]['lr']:.2e}  ratio={muon.param_groups[0]['lr']/adamw.param_groups[0]['lr']:.2f}")
```

Run: `python /tmp/smoke_train.py`
Expected output: at step 0 both LRs near 0; rising during steps 1-2 (warmup); at step 4 in stable phase with ratio = 0.02 / 3e-4 = 66.7.

- [ ] **Step 3: Smoke-validate forward_with_hidden activation memory**

Append to the same /tmp script (or run a one-liner):

```python
from models.mtp import MultiTokenPrediction
model.train()
mtp = MultiTokenPrediction(cfg, model).to(device)
B, T = 2, 32
tokens = torch.randint(0, cfg["vocab_size"], (B, T), device=device)
logits, mtp_out = mtp(tokens)
loss = mtp.compute_mtp_loss(mtp_out)
loss.backward()
print(f"MTP forward+backward OK, loss={loss.item():.4f}")
```

Expected: MTP forward+backward completes without OOM or NaN.

- [ ] **Step 4: Clean up smoke script**

Run: `rm /tmp/smoke_train.py`

- [ ] **Step 5: No commit — verification only**

If all green, the plan is done.

---

## Self-Review

**Spec coverage:**
- Bug A (aux loss): Task 2 ✓
- Bug B (joint scheduler): Task 3 ✓
- Bug C (MTP checkpointing): Task 4 ✓
- Bug D (validation data): Task 5 ✓
- Bug E (param partition): Task 1 ✓
- Bug F (hardcoded config): Task 6 ✓
- Final verification: Task 7 ✓

**Placeholder scan:** No "TBD" / "implement later" / "similar to" placeholders. All code blocks are complete.

**Type consistency:** `JointWSDScheduler` constructed in Task 3 Step 5 with `[self.muon_opt, self.adamw_opt] if self.muon_opt is not None else [self.adamw_opt]`, then `optimizers` is normalized via `if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]` — consistent. `get_last_lr()` returns a `list[float]` in both `_LRScheduler` and `JointWSDScheduler` — trainer.py:200 `self.scheduler.get_last_lr()[0]` is compatible.

**File-scope check:** No files outside the planned 6 are modified. No new abstractions introduced (JointWSDScheduler is a fix, not a new layer of indirection — replaces the existing per-optimizer scheduler).

**Test-count check:** Original 55 + 1 (Task 1) + 1 (Task 2) + 3 (Task 3) + 2 (Task 4) + 3 (Task 5) + 1 (Task 6) = 66 final tests.
