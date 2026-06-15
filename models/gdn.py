# models/gdn.py
"""Gated Delta Net (GDN) — Linear Attention via Delta Rule."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    """Gated Delta Net block."""

    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config["dim"]
        self.d_inner = config["gdn_d_inner"]
        self.d_state = config["gdn_d_state"]
        self.d_conv = config["gdn_d_conv"]
        self.headdim = config["gdn_headdim"]
        self.n_heads = self.d_inner // self.headdim
        self.chunk_size = config["gdn_chunk_size"]

        self.in_proj = nn.Linear(self.d_model, 6 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv, groups=self.d_inner, padding=self.d_conv - 1, bias=False)

        A_init = torch.arange(1, self.n_heads + 1, dtype=torch.float32).repeat_interleave(self.d_state).view(self.n_heads, self.d_state)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads).uniform_(0.001, 0.1))
        self.dt_bias._no_weight_decay = True

        self.b_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=False)
        self.g_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_heads, headdim, d_state = self.n_heads, self.headdim, self.d_state
        zxbcdtg = self.in_proj(x)
        z, x_in = zxbcdtg[..., 0*self.d_inner:1*self.d_inner], zxbcdtg[..., 1*self.d_inner:2*self.d_inner]
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        B_proj = self.b_proj(x_conv).view(B, T, n_heads, d_state)
        C_proj = self.c_proj(x_conv).view(B, T, n_heads, d_state)
        dt = F.softplus(self.dt_proj(x_conv) + self.dt_bias)
        g = torch.sigmoid(self.g_proj(x_conv))
        v = x_conv.view(B, T, n_heads, headdim)
        A = -torch.exp(self.A_log)
        y = self._chunked_delta_rule(v, dt, A, B_proj, C_proj)
        y = y + v * self.D.view(1, 1, n_heads, 1)
        y = y.reshape(B, T, self.d_inner)
        return self.out_proj(y * g * F.silu(z))

    def _chunked_delta_rule(self, v: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Chunked delta-rule recurrence (FP32 state)."""
        B_sz, T, n_heads, headdim = v.shape
        d_state = B.size(-1)
        chunk_size = self.chunk_size
        device, dtype = v.device, v.dtype
        decay = torch.sigmoid(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        k = F.normalize(B.float(), dim=-1, eps=1e-6).to(dtype)
        y = torch.empty(B_sz, T, n_heads, headdim, device=device, dtype=torch.float32)
        state = v.new_zeros(B_sz, n_heads, headdim, d_state, dtype=torch.float32)
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            v_chunk = v[:, chunk_start:chunk_end].float()
            k_chunk = k[:, chunk_start:chunk_end].float()
            decay_chunk = decay[:, chunk_start:chunk_end].float()
            for t in range(chunk_end - chunk_start):
                k_t, v_t, dec_t = k_chunk[:, t], v_chunk[:, t], decay_chunk[:, t]
                write = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
                state = dec_t.unsqueeze(-2) * state + write
                y[:, chunk_start + t] = (state @ C[:, chunk_start + t].float().unsqueeze(-1)).squeeze(-1)
        return y.to(dtype)
