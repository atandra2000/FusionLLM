# models/mamba.py
"""Mamba-2 selective state-space block for the hybrid schedule.

We implement Mamba-2 (Dao & Gu 2024) at a level the model code can
call directly.  The block is a drop-in replacement for the
``attn`` slot of a ``TransformerBlock`` when the layer schedule says
"this is an SSM layer".

Reference: ``Transformers are SSMs`` (arXiv 2405.21060) — the SSD
framework shows Mamba-2's selective SSM is a constrained structured
matrix multiplication.  We use the standard chunked parallel scan
implementation, parameterised by:

* ``d_model``        — input/output dim
* ``d_state``        — SSM state size N  (Mamba-2 default 128)
* ``d_conv``         — local conv width before the SSM  (default 4)
* ``d_inner``        — expanded dim  (defaults to 2 * d_model)
* ``headdim``        — per-head dim; ``d_inner`` is split into ``n_heads`` groups

We stay in pure PyTorch (no Triton kernel) so the code is portable;
the model trainer benchmarks and the user can swap in the optimised
``mamba_ssm`` CUDA kernel later without changing the layer interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    """One Mamba-2 block.  Drop-in for the attention slot in a layer."""

    def __init__(self, config: dict, layer_idx: int = 0, world_size: int = 1, rank: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.world_size = world_size
        self.rank = rank

        self.d_model = config["dim"]
        self.d_state = config.get("mamba_d_state", 128)
        self.d_conv = config.get("mamba_d_conv", 4)
        # Round up to nearest multiple of 8 for SSM kernel alignment.
        self.d_inner = int(2 * self.d_model)
        self.d_inner = (self.d_inner + 7) // 8 * 8
        self.headdim = config.get("mamba_headdim", 64)
        if self.d_inner % self.headdim != 0:
            self.d_inner = ((self.d_inner // self.headdim) + 1) * self.headdim
        self.n_heads = self.d_inner // self.headdim

        # Projections.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        # depthwise conv over the sequence dim, applied to the gated branch
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        # A_log: shape (n_heads, d_state); D: shape (n_heads,).
        # A is initialised to a small negative log so exp(A) is small.
        A = torch.arange(1, self.n_heads + 1, dtype=torch.float32).unsqueeze(-1).repeat(1, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True
        # dt projection: produces a per-token, per-head step size
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.dt_bias._no_weight_decay = True

        # B, C, dt projections: from d_inner to (n_heads * d_state) for B/C,
        # and n_heads for dt.
        self.b_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bsz, seqlen, d_model)  →  (bsz, seqlen, d_model)."""
        bsz, seqlen, _ = x.shape

        zxbcdt = self.in_proj(x)  # (bsz, seqlen, 2*d_inner)
        x_in, z = zxbcdt.chunk(2, dim=-1)  # (bsz, seqlen, d_inner) each

        # Local conv (causal)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Reshape x_conv to head format for selective scan
        x_conv_heads = x_conv.reshape(bsz, seqlen, self.n_heads, self.headdim)

        # Project to B, C, dt.
        B = self.b_proj(x_conv).view(bsz, seqlen, self.n_heads, self.d_state)
        C = self.c_proj(x_conv).view(bsz, seqlen, self.n_heads, self.d_state)
        dt = F.softplus(self.dt_proj(x_conv) + self.dt_bias)  # (bsz, seqlen, n_heads)

        A = -torch.exp(self.A_log)  # (n_heads, d_state)

        # Selective scan in pure PyTorch (slow but portable).
        y = self._selective_scan(x_conv_heads, dt, A, B, C, self.D)

        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,  # (bsz, seqlen, n_heads, headdim)
        dt: torch.Tensor,  # (bsz, seqlen, n_heads)
        A: torch.Tensor,  # (n_heads, d_state)
        B: torch.Tensor,  # (bsz, seqlen, n_heads, d_state)
        C: torch.Tensor,  # (bsz, seqlen, n_heads, d_state)
        D: torch.Tensor,  # (n_heads,)
    ) -> torch.Tensor:
        """Reference implementation of the Mamba-2 SSD scan.

        Computes, for each head, a per-token, per-state recurrence
            h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t
            y_t = C_t · h_t
        A is negative so the term is contractive.
        """
        bsz, seqlen, n_heads, headdim = x.shape
        d_state = B.size(-1)

        # Discretise A per token.
        dt_unsq = dt.unsqueeze(-1)  # (b, t, h, 1)
        A_d = dt_unsq * A.unsqueeze(0).unsqueeze(0)  # (b, t, h, d_state)
        A_bar = torch.exp(A_d)  # (b, t, h, d_state)

        # Recurrence.  We accumulate h in fp32 for stability.
        h = x.new_zeros(bsz, n_heads, headdim, d_state, dtype=torch.float32)
        ys = []
        for t in range(seqlen):
            x_t = x[:, t].to(torch.float32)  # (b, h, p)
            B_t = B[:, t].to(torch.float32)  # (b, h, d_state)
            # dt[:, t]: (b, h) -> unsqueeze(-1).unsqueeze(-1) -> (b, h, 1, 1)
            # B_t.unsqueeze(-2): (b, h, 1, d_state)
            # x_t.unsqueeze(-1): (b, h, p, 1)
            dt_broadcast = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            A_bar_t = A_bar[:, t].unsqueeze(-2)  # (b, h, 1, d_state)
            
            term1 = A_bar_t * h
            term2 = dt_broadcast * B_t.unsqueeze(-2) * x_t.unsqueeze(-1)
            h = term1 + term2
            C_t = C[:, t].to(torch.float32)  # (b, h, d_state)
            y_t = (C_t.unsqueeze(-2) * h).sum(dim=-1)  # (b, h, p)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (b, t, h, p)

        # Skip connection D * x (per head).
        y = y + x * D.view(1, 1, -1, 1)
        return y.view(bsz, seqlen, n_heads * headdim).to(x.dtype)
