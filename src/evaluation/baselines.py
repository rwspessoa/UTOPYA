"""
External baselines for anomaly detection on the batch distillation dataset.

All four baselines use the same sliding-window representation as UTOPYA
(W=120, stride=30) and are trained on the training split only, with
reconstruction/anomaly score evaluated on the test split.

Baselines (paper Section 5.3):
  1. PCA                    – linear reconstruction; Hotelling's T² score
  2. Feedforward Autoencoder – 2-layer MLP encoder-decoder; MSE score
  3. Isolation Forest        – ensemble anomaly score
  4. LSTM Autoencoder        – seq2seq reconstruction; mean-timestep MSE

PCA/FF-AE/Isolation Forest are fit on per-window *statistical summary
features* (B, N_vars*5: mean, std, min, max, linear slope per variable),
while the LSTM Autoencoder is fit on raw sequential data (B, W, N_vars).

Each baseline exposes .fit(X_train) and .score(X) → scores (N,) higher=more anomalous.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helper: flatten windows
# ---------------------------------------------------------------------------

def _flatten(X: np.ndarray) -> np.ndarray:
    """(N, W, V) → (N, W*V)"""
    return X.reshape(len(X), -1)


def _summary_features(X: np.ndarray) -> np.ndarray:
    """(N, W, V) -> (N, V*5): per-variable [mean, std, min, max, linear_slope] over the window's time axis."""
    N, W, V = X.shape
    mean = X.mean(axis=1)                      # (N, V)
    std = X.std(axis=1)                         # (N, V)
    vmin = X.min(axis=1)                         # (N, V)
    vmax = X.max(axis=1)                         # (N, V)
    t = np.arange(W, dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered ** 2).sum()
    # slope via least-squares closed form, vectorized over (N, V): cov(t, x) / var(t)
    x_centered = X - X.mean(axis=1, keepdims=True)             # (N, W, V)
    slope = np.tensordot(x_centered, t_centered, axes=([1], [0])) / denom   # (N, V)
    return np.concatenate([mean, std, vmin, vmax, slope], axis=1).astype(np.float32)   # (N, V*5)


# ---------------------------------------------------------------------------
# 1. PCA baseline
# ---------------------------------------------------------------------------

class PCABaseline:
    """
    Principal Component Analysis trained on normal/all training windows.
    Anomaly score: Hotelling's T² — squared Mahalanobis distance in PCA space,
    equivalently the reconstruction error in the original space.
    """

    def __init__(self, n_components: float = 0.95):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.pca     = PCA(n_components=n_components, svd_solver="full")
        self.scaler  = StandardScaler()
        self.n_components = n_components

    def fit(self, X: np.ndarray):
        """X: (N, W, V)"""
        Xf = _summary_features(X)
        Xs = self.scaler.fit_transform(Xf)
        self.pca.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns combined Hotelling's T² + SPE score (N,) — higher = more anomalous."""
        Xf = _summary_features(X)
        Xs = self.scaler.transform(Xf)
        scores_pca = self.pca.transform(Xs)
        t2 = (scores_pca ** 2 / self.pca.explained_variance_).sum(axis=1)
        Xr = self.pca.inverse_transform(scores_pca)
        spe = np.mean((Xs - Xr) ** 2, axis=1)
        t2_norm = (t2 - t2.mean()) / (t2.std() + 1e-8)
        spe_norm = (spe - spe.mean()) / (spe.std() + 1e-8)
        return t2_norm + spe_norm


# ---------------------------------------------------------------------------
# 2. Feedforward Autoencoder baseline
# ---------------------------------------------------------------------------

class _FFAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FFAutoencoderBaseline:
    """Feedforward autoencoder; MSE reconstruction score."""

    def __init__(
        self,
        latent_dim: int = 64,
        n_epochs:   int = 50,
        batch_size: int = 256,
        lr:         float = 1e-3,
        device:     str = "cuda",
    ):
        self.latent_dim = latent_dim
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = device
        self.model: Optional[_FFAE] = None
        self._in_dim: int = 0

    def fit(self, X: np.ndarray):
        Xf = _summary_features(X).astype(np.float32)
        self._in_dim = Xf.shape[1]
        self.model   = _FFAE(self._in_dim, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ds  = TensorDataset(torch.from_numpy(Xf))
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for ep in range(self.n_epochs):
            for (xb,) in dl:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(xb), xb)
                loss.backward()
                opt.step()
            if (ep + 1) % 10 == 0:
                print(f"    [FF-AE epoch {ep+1}/{self.n_epochs}]")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xf = _summary_features(X).astype(np.float32)
        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(Xf), self.batch_size):
                xb = torch.from_numpy(Xf[i:i + self.batch_size]).to(self.device)
                xr = self.model(xb)
                mse = nn.functional.mse_loss(xr, xb, reduction="none").mean(dim=1)
                scores.append(mse.cpu().numpy())
        return np.concatenate(scores)


# ---------------------------------------------------------------------------
# 3. Isolation Forest baseline
# ---------------------------------------------------------------------------

class IsolationForestBaseline:
    """Isolation Forest anomaly score (negated; higher = more anomalous)."""

    def __init__(self, n_estimators: int = 200, contamination: float = 0.15):
        from sklearn.ensemble import IsolationForest
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray):
        self.clf.fit(_summary_features(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal → negate for anomaly score
        return -self.clf.decision_function(_summary_features(X))


# ---------------------------------------------------------------------------
# 4. LSTM Autoencoder baseline
# ---------------------------------------------------------------------------

class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_vars: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.encoder  = nn.LSTM(n_vars,  hidden, n_layers, batch_first=True)
        self.decoder  = nn.LSTM(hidden,  n_vars, n_layers, batch_first=True)
        # Project encoder final hidden to decoder initial hidden (shape-compatible)
        self.h_proj   = nn.Linear(hidden, n_vars)
        self.hidden   = hidden
        self.n_layers = n_layers
        self.n_vars   = n_vars

    def forward(self, x):
        # x: (B, T, n_vars)
        B, T, V = x.shape
        _, (h_enc, _) = self.encoder(x)   # h_enc: (n_layers, B, hidden)

        # Project encoder hidden to decoder hidden size (n_vars)
        h_dec = self.h_proj(h_enc)         # (n_layers, B, n_vars)
        c_dec = torch.zeros_like(h_dec)

        # Decoder input: repeat projected latent over time
        dec_in = h_enc[-1].unsqueeze(1).expand(-1, T, -1)   # (B, T, hidden)
        out, _ = self.decoder(dec_in, (h_dec, c_dec))
        return out   # (B, T, n_vars)


class LSTMAutoencoderBaseline:
    """LSTM seq2seq autoencoder; per-timestep MSE averaged over time."""

    def __init__(
        self,
        hidden:     int = 64,
        n_layers:   int = 2,
        n_epochs:   int = 30,
        batch_size: int = 128,
        lr:         float = 1e-3,
        device:     str = "cuda",
    ):
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = device
        self.model: Optional[_LSTMAutoencoder] = None

    def fit(self, X: np.ndarray):
        """X: (N, W, V)"""
        Xt = torch.from_numpy(X.astype(np.float32))
        n_vars = X.shape[2]
        self.model = _LSTMAutoencoder(n_vars, self.hidden, self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ds  = TensorDataset(Xt)
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for ep in range(self.n_epochs):
            for (xb,) in dl:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(xb), xb)
                loss.backward()
                opt.step()
            if (ep + 1) % 10 == 0:
                print(f"    [LSTM-AE epoch {ep+1}/{self.n_epochs}]")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(X.astype(np.float32))
        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                xb = Xt[i:i + self.batch_size].to(self.device)
                xr = self.model(xb)
                mse = nn.functional.mse_loss(xr, xb, reduction="none").mean(dim=(1, 2))
                scores.append(mse.cpu().numpy())
        return np.concatenate(scores)
