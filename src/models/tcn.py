"""
Temporal Convolutional Network (TCN) encoder.

Architecture (Section 3.2 of UTOPYA paper):
- L = 6 residual blocks
- Kernel size k = 3, dilation d_l = 2^(l-1) for l in {1,...,6}
- Receptive field R = 1 + 2*(k-1)*sum(2^l, l=0..L-1) = 127 timesteps
- Weight normalisation, ReLU, dropout p = 0.5
- Skip connections with dimension-matching projection when needed
- Global average pooling → z_ts ∈ R^{dmodel}
- Also returns per-timestep feature map Z_ts ∈ R^{T x dmodel}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    """Causal dilated 1-D convolution with left-padding only."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation  # left pad only
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv.weight_g, mean=1.0, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.dropout(F.relu(self.conv(x)))


class _TCNBlock(nn.Module):
    """
    One TCN residual block (single causal convolution + skip connection).

    Paper: "each containing a causal 1-D convolution with kernel size k=3 and
    dilation factor d_l = 2^(l-1)", giving receptive field R = 1 + (k-1)*sum(d_l).

    Equation 1 from the paper:
        h_l(t) = ReLU(sum_j W_l(j) * h_{l-1}(t - j*d_l) + b_l)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = _CausalConv1d(in_ch, out_ch, kernel, dilation, dropout)
        # Dimension-matching skip connection
        self.skip = (
            weight_norm(nn.Conv1d(in_ch, out_ch, 1))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.relu(self.conv(x) + self.skip(x))


class TCNEncoder(nn.Module):
    """
    TCN encoder producing a pooled embedding and a per-timestep feature map.

    Parameters
    ----------
    n_input_vars : int
        Number of input time-series variables (default 31).
    dmodel : int
        Shared embedding dimension (default 128).
    n_layers : int
        Number of residual blocks (default 6).
    kernel_size : int
        Convolution kernel size (default 3).
    dropout : float
        Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_input_vars: int = 31,
        dmodel: int = 128,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.dmodel = dmodel
        self.n_layers = n_layers

        # Linear projection from raw input to dmodel channels
        self.input_proj = nn.Linear(n_input_vars, dmodel)

        # Build L residual blocks with exponentially growing dilation
        blocks = []
        for l in range(n_layers):
            dilation = 2 ** l
            blocks.append(_TCNBlock(dmodel, dmodel, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, T, V)
            Input window.

        Returns
        -------
        z_ts : torch.Tensor, shape (B, dmodel)
            Pooled (global-average) embedding.
        Z_ts : torch.Tensor, shape (B, T, dmodel)
            Per-timestep feature map.
        """
        # Project input: (B, T, V) → (B, T, dmodel)
        h = self.input_proj(x)

        # TCN expects (B, C, T) — channels first
        h = h.permute(0, 2, 1)

        for block in self.blocks:
            h = block(h)

        # h: (B, dmodel, T)
        Z_ts = h.permute(0, 2, 1)           # (B, T, dmodel)
        z_ts = Z_ts.mean(dim=1)             # (B, dmodel) — global average pool

        return z_ts, Z_ts

    def receptive_field(self) -> int:
        """Theoretical receptive field of the TCN.

        R = 1 + (k-1) * sum(d_l, l=0..L-1)  [single conv per block]
        With k=3, L=6: R = 1 + 2*(2^6-1) = 127  (matches paper Section 3.2)
        """
        k = 3
        return 1 + (k - 1) * sum(2 ** l for l in range(self.n_layers))
