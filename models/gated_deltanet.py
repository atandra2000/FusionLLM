# models/gated_deltanet.py
"""Gated Delta Net (GDN) — drop-in replacement for ``Mamba2Block``.

GDN (Qwen3-Next style) is a linear-attention variant that replaces the
Mamba-2 selective scan with a **delta-rule** recurrence.  The forward
contract is identical to ``Mamba2Block``:

    forward(x)  →  y      # (bsz, seqlen, d_model) → same

so the hybrid ``TransformerBlock`` can dispatch by config flag
(``ssm_type: "gdn" | "mamba2"``).

Architecture
------------
* **Input projection** to ``(z, x, b, c, dt, g)`` — six branches:
  - ``z``  : output gate (post-SwiGLU)
  - ``x``  : the value stream
  - ``b, c`` : per-token keys/values for the delta rule
  - ``dt`` : step size
  - ``g``  : per-channel gate (output of the conv1d branch)
* **Conv1d (kernel=4)** depth-wise over ``x`` (causal).
* **Δ-rule (chunked)** — state update ``h_t = g_t · h_{t-1} + (k_t v_t^T)``
  with a *forget* factor ``g`` derived from the input projection and a
  *key* ``k_t = (b_t · x_t) / ||x_t||`` (the delta rule's normalised
  outer-product update).  We do this in **pure PyTorch** so the path is
  portable; a Triton kernel lives in :mod:`kernels.delta_rule` for
  production.

References
----------
* Yang et al., "Gated DeltaNet: Sequence Modeling with a Linear
  Time-Complexity Recurrent Network" (Qwen3-Next technical report,
  2025).
* Schlag et al., "Linear Transformers with State-Space Layers" (the
  delta-rule paper, arXiv 2406.03428).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    """One GDN block.  Drop-in for the attention slot in a layer.

    Config keys (all optional except ``dim``)
    -----------------------------------------
    * ``dim``              — model dim (required)
    * ``gdn_d_state``      — SSM/key state size (default 128)
    * ``gdn_d_conv``       — local conv width (default 4)
    * ``gdn_headdim``      — per-head dim (default 64)
    """

    def __init__(self, config: dict, layer_idx: int = 0, world_size: int = 1, rank: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.world_size = world_size
        self.rank = rank

        self.d_model = config["dim"]
        self.d_state = config.get("gdn_d_state", 128)
        self.d_conv = config.get("gdn_d_conv", 4)
        # Round d_inner up to multiple of headdim.
        self.d_inner = max(2 * self.d_model, 8)
        self.headdim = config.get("gdn_headdim", 64)
        if self.d_inner % self.headdim != 0:
            self.d_inner = ((self.d_inner // self.headdim) + 1) * self.headdim
        self.n_heads = self.d_inner // self.headdim

        # Project to (z, x, b, c, dt, g). 6 branches of d_inner each.
        self.in_proj = nn.Linear(self.d_model, 6 * self.d_inner, bias=False)

        # Causal depthwise conv over the value stream.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )

        # Per-head, per-state log-decay: stored as A_log, sign-flipped
        # in forward so that exp(-A_log) is the per-step decay (≤ 1).
        # Shape (n_heads, d_state); the inner sum in the recurrence
        # sweeps over d_state.
        A = (
            torch.arange(1, self.n_heads + 1, dtype=torch.float32)
            .repeat_interleave(self.d_state)
            .view(self.n_heads, self.d_state)
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Per-head D skip-connection (added to the output of each head).
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True

        # dt bias (per-head)
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.dt_bias._no_weight_decay = True

        # B, C, dt, g projections
        self.b_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)
        # g (per-channel gate after conv1d)
        self.g_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bsz, seqlen, d_model)  →  (bsz, seqlen, d_model)."""
        bsz, seqlen, _ = x.shape
        d_inner = self.d_inner
        n_heads = self.n_heads
        headdim = self.headdim
        d_state = self.d_state

        # 1) Project to 6 streams.
        zxbcdtg = self.in_proj(x)  # (b, t, 6 * d_inner)
        z = zxbcdtg[..., 0 * d_inner : 1 * d_inner]
        x_in = zxbcdtg[..., 1 * d_inner : 2 * d_inner]
        # b, c, dt, g are computed from the *post-conv* value stream.
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 2) Per-token B/C/dt and the post-conv gate g.
        B = self.b_proj(x_conv).view(bsz, seqlen, n_heads, d_state)
        C = self.c_proj(x_conv).view(bsz, seqlen, n_heads, d_state)
        dt = F.softplus(self.dt_proj(x_conv) + self.dt_bias)  # (b, t, h)
        g = torch.sigmoid(self.g_proj(x_conv))  # (b, t, d_inner)

        # 3) Per-head value stream.
        v = x_conv.view(bsz, seqlen, n_heads, headdim)

        # 4) Δ-rule recurrence (Triton fast-path with PyTorch fallback).
        A = -torch.exp(self.A_log)  # (h, d_state), ≤ -1
        try:
            from kernels.delta_rule import chunked_delta_rule, has_triton

            if has_triton():
                y = chunked_delta_rule(v, dt, A, B, C)
            else:
                # Use pure PyTorch chunked implementation as fallback
                y = chunked_delta_rule(v, dt, A, B, C)
        except (NotImplementedError, RuntimeError) as e:
            raise RuntimeError(
                "Delta-rule kernel failed. Install triton>=3.2.0 for optimal performance, "
                "or rely on the PyTorch fallback (slower but functional)."
            ) from e

        # 5) Per-head D skip + reshape.
        y = y + v * self.D.view(1, 1, -1, 1)
        y = y.reshape(bsz, seqlen, n_heads * headdim)

        # 6) Channel-wise gate g and output gate z (SwiGLU), then out_proj.
        y = y * g * F.silu(z)
        return self.out_proj(y)

     # ── Δ-rule (chunked) ──────────────────────────────────────────────
    def _delta_rule(
        self,
        v: torch.Tensor,  # (b, t, h, p)
        dt: torch.Tensor,  # (b, t, h)
        A: torch.Tensor,  # (h, d_state)
        B: torch.Tensor,  # (b, t, h, d_state)
        C: torch.Tensor,  # (b, t, h, d_state)
    ) -> torch.Tensor:
        """Reference delta-rule scan in pure PyTorch.

        For each head h, the state h_state ∈ R^{p × d_state} updates as::

            decay_t = sigmoid(softplus(dt_t) · A_h)        # (h, d_state)
            k_t     = B_t / (||B_t||_2 + eps)              # (h, d_state)  — normalised
            h_state = decay_t · h_state + (k_t ⊗ v_t)     # outer product
            y_t     = C_t · h_state                         # (p,)

        We accumulate in fp32 for stability and project the state back
        to the input dtype at the end.
        """
        bsz, seqlen, n_heads, headdim = v.shape
        d_state = B.size(-1)

        # Per-step decay.  softplus keeps dt positive; A is ≤ -1 so the
        # product is negative and the sigmoid clamps it into (0, 1).
        decay = torch.sigmoid(F.softplus(dt).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Run the recurrence token-by-token (this is the *reference*
        # path; the Triton kernel does the chunked version).
        # Preallocate output tensor to avoid list append overhead + huge activation graph.
        y = torch.empty(bsz, seqlen, n_heads, headdim, device=v.device, dtype=torch.float32)
        state = v.new_zeros(bsz, n_heads, headdim, d_state, dtype=torch.float32)
        for t in range(seqlen):
            k_t = F.normalize(B[:, t].to(torch.float32), dim=-1, eps=1e-6)
            v_t = v[:, t].to(torch.float32)
            # decay[..., t] : (b, h, d_state) → broadcast to (b, h, p, d_state)
            state = decay[:, t].unsqueeze(-2) * state + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            c_t = C[:, t].to(torch.float32)  # (b, h, d_state)
            y[:, t] = (c_t.unsqueeze(-2) * state).sum(dim=-1)  # (b, h, p)
        return y.to(v.dtype)
