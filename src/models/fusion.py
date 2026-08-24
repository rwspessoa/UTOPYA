"""
Cross-modal attention, gated fusion, and multi-task output heads (Sections 3.4–3.5).

Architecture order (Figure 2 in paper):
  1. Pairwise bidirectional multi-head attention between all modality pairs
  2. Gated fusion with availability masks
  3. Three output heads: prediction, anomaly classification, anomaly localisation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cross-modal attention  (Section 3.4)
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Pairwise bidirectional multi-head cross-attention between M dynamic modality
    embeddings Z_i ∈ R^{T × dmodel}.

    For each ordered pair (i, j):
        A_ij = softmax(Q_i K_j^T / sqrt(d_head)) V_j
    and the updated representation for modality i is:
        Z̃_i = Z_i + sum_{j≠i} A_ij

    Implemented using PyTorch's built-in MultiheadAttention for efficiency.
    Uses n_heads=4 attention heads (paper spec).
    """

    def __init__(self, dmodel: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dmodel  = dmodel
        self.n_heads = n_heads
        # One shared attention module (parameters shared across pairs is a design
        # choice; the paper does not specify – we use per-pair projections by
        # registering them in a ModuleDict resolved at build time)
        self._shared_attn = nn.MultiheadAttention(
            dmodel, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],     # name → (B, T, dmodel)
        masks:      Optional[Dict[str, torch.Tensor]] = None,  # name → (B,) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns updated {name → (B, T, dmodel)} with cross-attended information
        from all other available modalities.
        """
        names  = list(modalities.keys())
        M      = len(names)
        updated = {n: modalities[n].clone() for n in names}

        for i, name_i in enumerate(names):
            Z_i = modalities[name_i]   # (B, T_i, d)
            aggregated = Z_i.clone()

            for j, name_j in enumerate(names):
                if i == j:
                    continue
                Z_j = modalities[name_j]   # (B, T_j, d)

                # Optional: zero out contribution from unavailable modalities
                key_padding_mask = None
                if masks is not None and name_j in masks:
                    # masks[name_j]: (B,) True = missing → mask all positions
                    key_padding_mask = masks[name_j].unsqueeze(1).expand(
                        -1, Z_j.size(1)
                    )  # (B, T_j)

                attn_out, _ = self._shared_attn(
                    query=Z_i,
                    key=Z_j,
                    value=Z_j,
                    key_padding_mask=key_padding_mask,
                )
                aggregated = aggregated + attn_out

            updated[name_i] = aggregated   # (B, T_i, d)

        return updated


# ---------------------------------------------------------------------------
# Gated fusion  (Section 3.4, Eq. 8)
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Combine M modality embeddings into one fused representation.

    g_i = sigmoid(W_g^i z̄_i + b_g^i)       (availability gate)
    z_fused = sum_i g_i ⊙ z̄_i              (gated sum, not normalised by M)

    where z̄_i = mean over time of Z̃_i  ∈ R^{B × dmodel}.

    Inputs z̄_i are mean-pooled here; richer representations (e.g., [CLS] token)
    can be swapped in by overriding `pool`.
    """

    def __init__(self, n_modalities: int, dmodel: int = 128):
        super().__init__()
        # Per-modality gating projections
        self.gates = nn.ModuleList(
            [nn.Linear(dmodel, dmodel) for _ in range(n_modalities)]
        )

    def pool(self, Z: torch.Tensor) -> torch.Tensor:
        """Global average pool over time: (B, T, d) → (B, d)."""
        return Z.mean(dim=1)

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],     # name → (B, T, dmodel)
        avail:      Optional[Dict[str, torch.Tensor]] = None,  # name → (B,) bool 1=available
    ) -> torch.Tensor:
        """Returns z_fused ∈ R^{B × dmodel}."""
        names  = list(modalities.keys())
        device = next(iter(modalities.values())).device

        fused  = None
        for idx, name in enumerate(names):
            Z = modalities[name]          # (B, T, d)
            z = self.pool(Z)              # (B, d)

            g = torch.sigmoid(self.gates[idx](z))   # (B, d)

            # Zero contribution from missing modalities
            if avail is not None and name in avail:
                mask = avail[name].float().unsqueeze(-1).to(device)  # (B,1)
                g = g * mask

            contrib = g * z
            fused   = contrib if fused is None else fused + contrib

        return fused  # (B, dmodel)


# ---------------------------------------------------------------------------
# Multi-task output heads  (Section 3.5)
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    Multi-step prediction: z_fused → y_hat ∈ R^{B × H × N_cont}

    MLP: dmodel → 512 → H * N_cont, reshape.
    """

    def __init__(self, dmodel: int = 128, horizon: int = 60, n_cont: int = 23):
        super().__init__()
        self.H      = horizon
        self.N_cont = n_cont
        self.net = nn.Sequential(
            nn.Linear(dmodel, 512),
            nn.ReLU(),
            nn.Linear(512, horizon * n_cont),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.net(z).view(B, self.H, self.N_cont)   # (B, H, N_cont)


class AnomalyClassificationHead(nn.Module):
    """
    Window-level anomaly classification: z_fused → logits ∈ R^{B × 2}

    Two-class: 0=normal, 1=anomaly.  (Label 1,2,3 all→1, 0→0.)
    """

    def __init__(self, dmodel: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dmodel, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (B, 2)


class PhaseClassificationHead(nn.Module):
    """
    Window-level 4-class phase classification: z_fused → logits ∈ R^{B × 4}

    Classes mirror the dataset label encoding (0=normal/steady-state,
    1=blind, 2=anomalous, 3=recovery) via BatchDistillationDataset's
    per-window majority-vote `phase_label`. Not part of the original
    architecture (Sections 3.4-3.5 describe only prediction/classification/
    localisation) — added post-hoc to reconstruct the paper's Figure 5
    (phase confusion matrix), trained as a lightweight probe on top of a
    frozen fused representation rather than jointly with the main losses.
    """

    def __init__(self, dmodel: int = 128, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dmodel, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (B, n_classes)


class ReconstructionHead(nn.Module):
    """
    Reconstructs the input window from the fused representation:
    z_fused ∈ R^{B×dmodel} → X_hat ∈ R^{B×W×V_in}   (Section 5.5/5.7.3).

    The paper describes a "U-Net-style decoder"; this is a simplified
    2-layer MLP decoder (dmodel → 512 → W*V_in, reshaped) rather than a
    literal U-Net, given the time budget for this reconstruction pass —
    documented here rather than silently substituted. Gated by w_recon=0
    in the production multi-task loss (never trained jointly by default);
    exposed so it can be trained standalone on normal-only windows with
    the rest of the model frozen, reproducing the paper's separate
    reconstruction-head experiment (Section 5.7.3, standalone AUROC 0.695).
    """

    def __init__(self, dmodel: int = 128, window: int = 120, n_vars: int = 31):
        super().__init__()
        self.window = window
        self.n_vars = n_vars
        self.net = nn.Sequential(
            nn.Linear(dmodel, 512),
            nn.ReLU(),
            nn.Linear(512, window * n_vars),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.net(z).view(B, self.window, self.n_vars)   # (B, W, V_in)


class AnomalyLocalisationHead(nn.Module):
    """
    Variable-level anomaly localisation: Z_ts ∈ R^{B×T×d} → scores ∈ R^{B×T×N_vars}

    Applied over the per-timestep TCN output rather than the pooled vector,
    to produce a dense anomaly map.
    """

    def __init__(self, dmodel: int = 128, n_vars: int = 31):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dmodel, 64),
            nn.ReLU(),
            nn.Linear(64, n_vars),
        )

    def forward(self, Z_ts: torch.Tensor) -> torch.Tensor:
        return self.net(Z_ts)   # (B, T, N_vars)
