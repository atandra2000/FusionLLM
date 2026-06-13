# models/gdn.py
"""Gated Delta Net (GDN) — Linear Attention via Delta Rule (Frozen v1 spec).

Architecture (per FINAL_FROZEN_SPEC.md §5.4):
  Input (B, T, 768)
    ├─ in_proj: Linear(768→6×1024=6144) → Split: z, x, b, c, dt, g
    ├─ x → Conv1d(1024, k=4, groups=1024, causal) → SiLU → x_conv
    ├─ b_proj: Linear(1024→32×32=1024)  →  B (B, T, 32, 32)
    ├─ c_proj: Linear(1024→32×32=1024)  →  C (B, T, 32, 32)
    ├─ dt_proj: Linear(1024→32) + dt_bias → SoftPlus → dt
    ├─ g_proj: Linear(1024→1024) → Sigmoid → g
    ├─ v = x_conv.view(B, T, 32, 32)     # headdim = 32
    ├─ A = -exp(A_log)   # (32, 32) fixed decay
    ├─ Delta-rule recurrence (chunked, pure PyTorch, chunk_size=64)
    ├─ y = y * g * SiLU(z)
    └─ out_proj: Linear(1024→768)

Per-layer params: 8,688,704
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    """Gated Delta Net block.

    Frozen v1 spec:
      - d_inner = 1024, d_state = 32, headdim = 32, n_heads = 32
      - d_conv = 4, chunk_size = 64
      - Pure PyTorch chunked delta-rule recurrence
      - No Triton
    """

    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config["dim"]                     # 768
        self.d_inner = config["gdn_d_inner"]             # 1024
        self.d_state = config["gdn_d_state"]             # 32
        self.d_conv = config["gdn_d_conv"]               # 4
        self.headdim = config["gdn_headdim"]             # 32
        self.n_heads = self.d_inner // self.headdim      # 32
        self.chunk_size = config["gdn_chunk_size"]       # 64

        # ── Input projection: (768 → 6 × 1024 = 6144) ─────────────────────
        self.in_proj = nn.Linear(self.d_model, 6 * self.d_inner, bias=False)

        # ── Causal depthwise conv1d over x stream ──────────────────────────
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,  # causal: pad left only
            bias=False,
        )

        # ── A_log (per-head, per-state log decay) ─────────────────────────
        # A = -exp(A_log) ≤ -1. Shape: (n_heads, d_state) = (32, 32)
        A_init = torch.arange(1, self.n_heads + 1, dtype=torch.float32).repeat_interleave(
            self.d_state
        ).view(self.n_heads, self.d_state)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True

        # ── D (per-head skip connection) ───────────────────────────────────
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True

        # ── dt_bias (per-head) ────────────────────────────────────────────
        self.dt_bias = nn.Parameter(
            torch.empty(self.n_heads).uniform_(0.001, 0.1)
        )
        self.dt_bias._no_weight_decay = True

        # ── B, C, dt, g projections ───────────────────────────────────────
        self.b_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)   # 1024→1024
        self.c_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)   # 1024→1024
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=False)                   # 1024→32 (dt_bias is separate)
        self.g_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)                    # 1024→1024

        # ── Output projection ──────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)    # 1024→768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        B, T, _ = x.shape
        d_inner = self.d_inner
        n_heads = self.n_heads
        headdim = self.headdim
        d_state = self.d_state

        # ── 1. Input projection → 6 streams ────────────────────────────────
        zxbcdtg = self.in_proj(x)  # (B, T, 6*d_inner)
        z = zxbcdtg[..., 0*d_inner : 1*d_inner]   # (B, T, d_inner)
        x_in = zxbcdtg[..., 1*d_inner : 2*d_inner]  # (B, T, d_inner)

        # ── 2. Causal conv1d on x stream ───────────────────────────────────
        # Conv1d expects (B, d_inner, T)
        x_conv = self.conv1d(x_in.transpose(1, 2))
        x_conv = x_conv[:, :, :T].transpose(1, 2)  # (B, T, d_inner), causal trim
        x_conv = F.silu(x_conv)

        # ── 3. Compute B, C, dt, g from post-conv x ────────────────────────
        B_proj = self.b_proj(x_conv).view(B, T, n_heads, d_state)  # (B, T, 32, 32)
        C_proj = self.c_proj(x_conv).view(B, T, n_heads, d_state)  # (B, T, 32, 32)
        dt = F.softplus(self.dt_proj(x_conv) + self.dt_bias)       # (B, T, n_heads)
        g = torch.sigmoid(self.g_proj(x_conv))                     # (B, T, d_inner)

        # ── 4. Value stream ────────────────────────────────────────────────
        v = x_conv.view(B, T, n_heads, headdim)  # (B, T, 32, 32)

        # ── 5. Decay matrix A ──────────────────────────────────────────────
        A = -torch.exp(self.A_log)  # (n_heads, d_state), ≤ -1

        # ── 6. Chunked delta-rule recurrence (pure PyTorch) ────────────────
        y = self._chunked_delta_rule(v, dt, A, B_proj, C_proj)

        # ── 7. Per-head D skip connection ──────────────────────────────────
        y = y + v * self.D.view(1, 1, n_heads, 1)  # (B, T, 32, 32)

        # ── 8. Flatten heads ───────────────────────────────────────────────
        y = y.reshape(B, T, d_inner)

        # ── 9. Gating: y * g * SiLU(z) ────────────────────────────────────
        y = y * g * F.silu(z)

        # ── 10. Output projection ──────────────────────────────────────────
        return self.out_proj(y)

    def _chunked_delta_rule(
        self,
        v: torch.Tensor,    # (B, T, n_heads, headdim)
        dt: torch.Tensor,   # (B, T, n_heads)
        A: torch.Tensor,    # (n_heads, d_state) — negative, ≤ -1
        B: torch.Tensor,    # (B, T, n_heads, d_state)
        C: torch.Tensor,    # (B, T, n_heads, d_state)
    ) -> torch.Tensor:
        """Chunked delta-rule recurrence in pure PyTorch.

        The delta rule state update at step t:
            decay_t = sigmoid(dt_t * A)     # (B, n_heads, d_state)
            k_t = normalize(B_t)            # (B, n_heads, d_state)
            state_t = decay_t * state_{t-1} + outer(k_t, v_t)
            y_t = C_t @ state_t

        We process in chunks of chunk_size tokens to balance numerical
        stability and performance. State is kept in FP32.
        """
        B_sz, T, n_heads, headdim = v.shape
        d_state = B.size(-1)
        chunk_size = self.chunk_size
        device = v.device
        dtype = v.dtype

        # Compute per-step decay: sigmoid(dt * A)
        # dt: (B, T, n_heads) → (B, T, n_heads, 1)
        # A: (n_heads, d_state) → (1, 1, n_heads, d_state)
        decay = torch.sigmoid(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, T, n_heads, d_state)

        # Normalize B to get keys
        k = F.normalize(B.float(), dim=-1, eps=1e-6).to(dtype)  # (B, T, n_heads, d_state)

        # Output buffer, accumulated in FP32
        y = torch.empty(B_sz, T, n_heads, headdim, device=device, dtype=torch.float32)

        # State: (B, n_heads, headdim, d_state), FP32
        state = v.new_zeros(B_sz, n_heads, headdim, d_state, dtype=torch.float32)

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            # Get chunk slices
            v_chunk = v[:, chunk_start:chunk_end].float()        # (B, chunk, n_heads, headdim)
            k_chunk = k[:, chunk_start:chunk_end].float()         # (B, chunk, n_heads, d_state)
            decay_chunk = decay[:, chunk_start:chunk_end].float()  # (B, chunk, n_heads, d_state)

            for t in range(chunk_len):
                k_t = k_chunk[:, t]      # (B, n_heads, d_state)
                v_t = v_chunk[:, t]      # (B, n_heads, headdim)
                dec_t = decay_chunk[:, t]  # (B, n_heads, d_state)

                # Write: state = decay * state + outer(k, v)
                # outer(k, v): (B, n_heads, d_state, 1) * (B, n_heads, 1, headdim)
                # → (B, n_heads, d_state, headdim) then transpose to (B, n_heads, headdim, d_state)
                k_unsq = k_t.unsqueeze(-2)    # (B, n_heads, 1, d_state)
                v_unsq = v_t.unsqueeze(-1)    # (B, n_heads, headdim, 1)
                write = v_unsq @ k_unsq        # (B, n_heads, headdim, d_state)

                # Apply decay and add outer product
                state = dec_t.unsqueeze(-2) * state + write

                # Read: y = C @ state
                c_t = C[:, chunk_start + t].float()  # (B, n_heads, d_state)
                # (B, n_heads, headdim, d_state) @ (B, n_heads, d_state, 1)
                # → (B, n_heads, headdim, 1)
                y_t = (state @ c_t.unsqueeze(-1)).squeeze(-1)  # (B, n_heads, headdim)
                y[:, chunk_start + t] = y_t

        return y.to(dtype)
