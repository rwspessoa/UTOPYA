"""
Static modality encoders and FiLM conditioning (Section 3.2–3.3).

Four static/dynamic encoders:
  - TabularEncoder  : numerical operating-point settings → z_tab ∈ R^{dmodel}
  - TextEncoder     : operation-log SBERT embeddings (384-d) → z_text ∈ R^{dmodel}
  - MolecularEncoder: per-component molecular graph (GCN K=3) → z_mol ∈ R^{dmodel}
  - AudioEncoder    : log-mel spectrogram 4-layer CNN → z_audio ∈ R^{dmodel}

Static encoders (tab, text, mol) are aggregated into FiLM context vector c:
  c = LN(ReLU(W_c [z_tab; z_text; z_mol] + b_c))   (Eq. 6)

FiLM conditioning (Eq. 7) drives the dynamic TCN embeddings.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tabular encoder  (Section 3.2 – "Experiment-level metadata")
# ---------------------------------------------------------------------------

class TabularEncoder(nn.Module):
    """
    Two-layer MLP: x_tab ∈ R^{d_tab} → z_tab ∈ R^{dmodel}

    Paper: W1 ∈ R^{256×dtab}, W2 ∈ R^{dmodel×256}, ReLU, dropout.
    """

    def __init__(self, d_tab: int, dmodel: int = 128, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_tab, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dmodel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, dmodel)


# ---------------------------------------------------------------------------
# Text encoder  (Section 3.2 – "Free-text fields")
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Projects pre-computed Sentence-BERT embeddings (384-d) to dmodel.

    z_text = W_text * SBERT(text) + b_text   (Eq. 4)

    Pre-computing SBERT offline and passing the 384-d vector as input avoids
    the overhead of running the language model inside the training loop.
    """

    def __init__(self, sbert_dim: int = 384, dmodel: int = 128):
        super().__init__()
        self.proj = nn.Linear(sbert_dim, dmodel)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)   # (B, dmodel)


# ---------------------------------------------------------------------------
# Audio encoder  (Section 3.2 – "Log-mel CNN for acoustic signal")
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Log-mel spectrogram → 4-layer CNN → global avg pool → Linear → z_audio ∈ R^{dmodel}

    Paper Section 3.2: raw waveform is converted to a log-mel spectrogram with
    n_mels=64 bins, then processed by a 4-layer CNN followed by global average
    pooling and a linear projection.

    Input: (B, 1, n_mels, T_audio)  log-mel spectrogram (single channel)
    """

    N_MELS = 64

    def __init__(self, n_mels: int = 64, dmodel: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            # Layer 1: 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 4: 128 → dmodel
            nn.Conv2d(128, dmodel, kernel_size=3, padding=1),
            nn.BatchNorm2d(dmodel),
            nn.ReLU(),
        )
        self.proj = nn.Linear(dmodel, dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, n_mels, T_audio)  log-mel spectrogram
        Returns z_audio ∈ R^{B × dmodel}
        """
        h = self.cnn(x)               # (B, dmodel, h', w')
        z = h.mean(dim=(-2, -1))      # global average pool → (B, dmodel)
        return self.proj(z)           # (B, dmodel)


# ---------------------------------------------------------------------------
# NMR encoder  (Section 5.9 – "NMR composition, ~1/min per-experiment summary")
# ---------------------------------------------------------------------------

class NMREncoder(nn.Module):
    """
    Small MLP over a fixed-size per-experiment NMR composition summary
    (mean mole fractions of each component + assumed-impurity fraction,
    see src/data/nmr.py) → z_nmr ∈ R^{dmodel}.

    Not part of the original 8-modality architecture description in
    Sections 3.2-3.5; added post-hoc to reconstruct Section 5.9's A15
    (frozen-backbone NMR extension). Treated as a per-experiment summary
    (constant across a window), the same simplification already used for
    AudioEncoder's input in this codebase.
    """

    def __init__(self, n_features: int = 4, dmodel: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dmodel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, dmodel)


# ---------------------------------------------------------------------------
# Image encoder  (Section 5.9 – "ResNet-18 backbone, 3 cameras, mean pooled")
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    Projects a precomputed, frozen-ResNet-18 (ImageNet-pretrained) 512-d
    feature vector — already mean-pooled across the 3 camera views at
    cache-build time (see src/data/image_cache.py) — to z_image ∈ R^{dmodel}.

    Mirrors TextEncoder's pattern of consuming a precomputed embedding
    rather than running the backbone inside the training loop: ResNet-18
    is frozen (matching the paper's use of it as a fixed feature extractor),
    so caching its output once is equivalent to running it live every batch
    but far cheaper, and keeps this class a simple linear projection.
    """

    def __init__(self, feature_dim: int = 512, dmodel: int = 128):
        super().__init__()
        self.proj = nn.Linear(feature_dim, dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, dmodel)


# ---------------------------------------------------------------------------
# Molecular encoder  (Section 3.2 – "GCN for molecular identity")
# ---------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    """One graph convolutional layer (Eq. 5, Kipf & Welling 2017)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        h        : (N, in_dim)   node features
        adj_norm : (N, N)        symmetrically-normalised adjacency (with self-loops)
        """
        return F.relu(self.linear(adj_norm @ h))


class MolecularEncoder(nn.Module):
    """
    K=3-layer GCN over atom/bond graph, followed by global mean pooling → z_mol ∈ R^{dmodel}.

    Node features: one-hot atomic number + degree + aromaticity + H count (16-d).
    Graphs for the three chemical components are pre-built and stored as buffers.
    """

    N_ATOM_FEATURES = 16
    K_LAYERS = 3

    def __init__(self, dmodel: int = 128, n_atom_features: int = 16):
        super().__init__()
        dims = [n_atom_features] + [dmodel] * self.K_LAYERS
        self.gcn_layers = nn.ModuleList(
            [_GCNLayer(dims[i], dims[i + 1]) for i in range(self.K_LAYERS)]
        )
        self.output_proj = nn.Linear(dmodel, dmodel)

    def forward(
        self,
        node_feats: torch.Tensor,      # (N_atoms, n_atom_features)
        adj_norm:   torch.Tensor,      # (N_atoms, N_atoms)
        batch_idx:  Optional[torch.Tensor] = None,  # not needed for single molecule
    ) -> torch.Tensor:
        """
        Returns z_mol ∈ R^{dmodel} for a single molecule (or batched via vmap).
        """
        h = node_feats
        for layer in self.gcn_layers:
            h = layer(h, adj_norm)           # (N, dmodel)
        z = h.mean(dim=0)                    # global mean pool → (dmodel,)
        return self.output_proj(z)


def build_molecule_graph(smiles: str, device: torch.device = torch.device("cpu")):
    """
    Build normalised adjacency and atom features for a SMILES string.
    Uses RDKit; falls back to a zero vector if RDKit is unavailable.

    Returns (node_feats, adj_norm) or None.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdmolops
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()

    # --- node features (16-d) ---
    ATOMIC_NUMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H C N O F P S Cl Br I
    feats = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        one_hot = [float(an == x) for x in ATOMIC_NUMS]  # 10
        one_hot += [float(an not in ATOMIC_NUMS)]          # other
        one_hot += [
            atom.GetDegree() / 4.0,
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs() / 4.0,
            float(atom.IsInRing()),
            float(atom.GetFormalCharge() != 0),
        ]  # 5 extra → total 16
        feats.append(one_hot)

    node_feats = torch.tensor(feats, dtype=torch.float32, device=device)

    # --- normalised adjacency with self-loops ---
    adj = torch.zeros(n, n, device=device)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0
    adj += torch.eye(n, device=device)  # self-loops

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    D = torch.diag(deg_inv_sqrt)
    adj_norm = D @ adj @ D  # symmetric normalisation

    return node_feats, adj_norm


# ---------------------------------------------------------------------------
# Static context aggregator  (Eq. 6)
# ---------------------------------------------------------------------------

class StaticContextAggregator(nn.Module):
    """
    c = LN(ReLU(W_c [z_tab; z_text; z_mol] + b_c))

    W_c ∈ R^{dmodel × 3*dmodel}
    """

    def __init__(self, dmodel: int = 128):
        super().__init__()
        self.proj = nn.Linear(3 * dmodel, dmodel)
        self.ln = nn.LayerNorm(dmodel)

    def forward(
        self,
        z_tab:  torch.Tensor,   # (B, dmodel)
        z_text: torch.Tensor,   # (B, dmodel)
        z_mol:  torch.Tensor,   # (B, dmodel)
    ) -> torch.Tensor:
        cat = torch.cat([z_tab, z_text, z_mol], dim=-1)  # (B, 3*dmodel)
        return self.ln(F.relu(self.proj(cat)))             # (B, dmodel)


# ---------------------------------------------------------------------------
# FiLM conditioner  (Section 3.3, Eq. 7)
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """
    Applies FiLM to a dynamic embedding z_i conditioned on context vector c.

    z'_i = γ_i ⊙ z_i + β_i
    where γ_i = W_γ c + b_γ  (init b_γ = 1 → identity at start)
          β_i = W_β c + b_β  (init b_β = 0)

    A separate FiLMConditioner is instantiated for each dynamic modality.
    """

    def __init__(self, dmodel: int = 128):
        super().__init__()
        self.gamma_proj = nn.Linear(dmodel, dmodel)
        self.beta_proj  = nn.Linear(dmodel, dmodel)
        self._init_weights()

    def _init_weights(self):
        # Identity initialisation: γ→1, β→0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(c)    # (B, dmodel)
        beta  = self.beta_proj(c)     # (B, dmodel)
        return gamma * z + beta       # element-wise (Eq. 7)
