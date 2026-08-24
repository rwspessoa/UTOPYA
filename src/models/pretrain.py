"""
Self-supervised pretraining for the TCN encoder (Section 4.2).

Two complementary objectives:
1. Block-masked reconstruction
   - Randomly zero contiguous segments of 10–30 timesteps
   - Train a lightweight decoder to reconstruct the masked positions
   - Forces the encoder to learn temporal dependencies beyond interpolation

2. Contrastive loss (NT-Xent style, following Yue et al. 2022)
   - Two augmented views of the same window are positive pairs
   - Windows from different experiments are negative pairs
   - Encourages similar representations for the same window,
     discriminative representations across windows
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn import TCNEncoder
from src.data.dataset import N_INPUT_VARS, WINDOW_SIZE


# ---------------------------------------------------------------------------
# Augmentations for contrastive pretraining
# ---------------------------------------------------------------------------

def _jitter(x: torch.Tensor, sigma_range: Tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    """Additive Gaussian noise with sigma sampled uniformly per call from
    sigma_range (paper: sigma in [0.01, 0.05]), rather than a single fixed
    sigma, so different augmented views see different noise magnitudes."""
    sigma = random.uniform(*sigma_range)
    return x + torch.randn_like(x) * sigma


def _scaling(x: torch.Tensor, lo: float = 0.9, hi: float = 1.1) -> torch.Tensor:
    scale = torch.empty(x.shape[0], 1, x.shape[2], device=x.device).uniform_(lo, hi)
    return x * scale


def _drop_segment(x: torch.Tensor, min_len: int = 10, max_len: int = 30) -> torch.Tensor:
    """Zero out a random contiguous segment (used for both masking and augmentation)."""
    x = x.clone()
    B, T, V = x.shape
    seg_len = random.randint(min_len, min(max_len, T // 2))
    start = random.randint(0, T - seg_len)
    x[:, start: start + seg_len, :] = 0.0
    return x, start, seg_len


def augment_view(x: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to produce a contrastive view."""
    if random.random() < 0.5:
        x = _jitter(x)
    if random.random() < 0.5:
        x = _scaling(x)
    return x


# ---------------------------------------------------------------------------
# Masked reconstruction decoder
# ---------------------------------------------------------------------------

class MaskReconDecoder(nn.Module):
    """
    Lightweight MLP decoder that reconstructs masked timesteps from the
    per-timestep feature map Z_ts.

    Input : Z_ts ∈ (B, T, dmodel)
    Output: x_hat ∈ (B, T, V)   — reconstruction of the full window
    """

    def __init__(self, dmodel: int = 128, n_input_vars: int = N_INPUT_VARS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, n_input_vars),
        )

    def forward(self, Z_ts: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z_ts)   # (B, T, V)


# ---------------------------------------------------------------------------
# Contrastive projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Small MLP projection head for NT-Xent contrastive loss."""

    def __init__(self, dmodel: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent loss for a batch of (z1, z2) positive pairs.

    z1, z2: (B, proj_dim), L2-normalised.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)            # (2B, proj_dim)
    sim = torch.mm(z, z.T) / temperature       # (2B, 2B)

    # Mask out self-similarities on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pair indices: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Pretraining model wrapper
# ---------------------------------------------------------------------------

class TCNPretrainer(nn.Module):
    """
    Wraps TCNEncoder with masked-reconstruction decoder and contrastive head
    for self-supervised pretraining.
    """

    def __init__(
        self,
        encoder: Optional[TCNEncoder] = None,
        dmodel: int = 128,
        n_input_vars: int = N_INPUT_VARS,
        proj_dim: int = 64,
        mask_min: int = 10,
        mask_max: int = 30,
        contrastive_temp: float = 0.1,
        lambda_recon: float = 1.0,
        lambda_contrast: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else TCNEncoder(
            n_input_vars=n_input_vars, dmodel=dmodel
        )
        self.decoder = MaskReconDecoder(dmodel=dmodel, n_input_vars=n_input_vars)
        self.proj_head = ProjectionHead(dmodel=dmodel, proj_dim=proj_dim)

        self.mask_min = mask_min
        self.mask_max = mask_max
        self.temp = contrastive_temp
        self.lambda_recon = lambda_recon
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, V)

        Returns
        -------
        loss_total, loss_recon, loss_contrast
        """
        B, T, V = x.shape
        device = x.device

        # ---- Block-masked reconstruction ----
        x_masked = x.clone()
        seg_len = random.randint(self.mask_min, min(self.mask_max, T // 2))
        mask_start = random.randint(0, T - seg_len)
        x_masked[:, mask_start: mask_start + seg_len, :] = 0.0

        _, Z_ts = self.encoder(x_masked)        # (B, T, dmodel)
        x_hat = self.decoder(Z_ts)              # (B, T, V)

        # Only compute reconstruction loss at masked positions
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[mask_start: mask_start + seg_len] = True
        loss_recon = F.mse_loss(x_hat[:, mask, :], x[:, mask, :])

        # ---- Contrastive loss ----
        view1 = augment_view(x)
        view2 = augment_view(x)

        z1, _ = self.encoder(view1)
        z2, _ = self.encoder(view2)

        p1 = self.proj_head(z1)
        p2 = self.proj_head(z2)

        loss_contrast = nt_xent_loss(p1, p2, temperature=self.temp)

        loss_total = self.lambda_recon * loss_recon + self.lambda_contrast * loss_contrast

        return loss_total, loss_recon, loss_contrast
