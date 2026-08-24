"""
PyTorch Dataset for the UTOPYA pipeline.

Implements:
- Per-experiment normalisation (subtract mean / divide std of first 300 timesteps)
- Sliding window segmentation: W=120, stride=30
- Prediction targets: H=60 future timesteps for 25 continuous variables
- Window-level anomaly label (majority vote) and phase label
- Curriculum difficulty scores

Window label encoding mirrors the dataset:
    0 – normal
    1 – blind
    2 – anomalous
    3 – recovery

Binary anomaly label: window is anomalous if ANY timestep has label > 0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import (
    ALL_COLS,
    BINARY_COLS,
    ExperimentMeta,
)

# ---------------------------------------------------------------------------
# Constants (from paper)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 120        # W
STRIDE = 30              # s
PRED_HORIZON = 60        # H
NORM_BASELINE_LEN = 300  # first 300 timesteps for per-experiment normalisation

# 25 continuous target variables = all columns except 6 binary valve/pump cols
BINARY_COL_INDICES = [ALL_COLS.index(c) for c in BINARY_COLS if c in ALL_COLS]
CONTINUOUS_COL_INDICES = [i for i in range(len(ALL_COLS)) if i not in BINARY_COL_INDICES]
N_CONTINUOUS = len(CONTINUOUS_COL_INDICES)   # should be 25 (31 - 6 binary)
N_INPUT_VARS = len(ALL_COLS)                 # 31

# Curriculum difficulty scores (Section 4.2)
DIFFICULTY = {0: 0.0, 1: 0.9, 2: 0.3, 3: 0.6}  # normal, blind, anomalous, recovery
# Mixed-phase windows (containing transitions) get 0.5; assigned at window build time


# ---------------------------------------------------------------------------
# Per-experiment normalisation
# ---------------------------------------------------------------------------

def normalise_experiment(data: np.ndarray, baseline_len: int = NORM_BASELINE_LEN) -> np.ndarray:
    """
    Subtract per-variable mean and divide by std computed over the first
    `baseline_len` timesteps.  Binary columns are left unchanged.

    Robust handling:
    - If baseline std < 1.0 (sensor too stable in startup), fall back to the
      global std of the full series so anomaly magnitudes remain comparable.
    - Final clip to [-10, 10] prevents gradient explosions while preserving
      the direction of large anomaly deviations.
    """
    data = data.copy()
    baseline = data[:min(baseline_len, len(data))]

    for i in range(data.shape[1]):
        if i in BINARY_COL_INDICES:
            continue
        mu    = float(baseline[:, i].mean())
        sigma = float(baseline[:, i].std())
        if sigma < 1.0:
            # Fallback: use std over the entire series
            sigma = float(data[:, i].std())
        if sigma < 1e-6:
            sigma = 1.0
        data[:, i] = np.clip((data[:, i] - mu) / sigma, -10.0, 10.0)

    return data


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def build_windows(
    data: np.ndarray,          # (T, V) float32
    labels: np.ndarray,        # (T,)   int8
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    pred_horizon: int = PRED_HORIZON,
) -> List[Dict]:
    """
    Slice a full experiment into overlapping windows.

    Each window dict contains:
      x          : (W, V)           input window
      y_target   : (H, N_CONT)      prediction targets (continuous vars only)
      label      : int               binary anomaly label (0/1)
      phase_label: int               majority phase label (0-3)
      difficulty : float             curriculum difficulty score
    """
    T = len(data)
    windows = []
    for start in range(0, T - window_size - pred_horizon + 1, stride):
        end = start + window_size
        future_end = end + pred_horizon

        x = data[start:end]                              # (W, V)
        y_raw = data[end:future_end]                     # (H, V)
        y_target = y_raw[:, CONTINUOUS_COL_INDICES]      # (H, N_CONT)

        window_labels = labels[start:end]
        unique_phases = set(window_labels.tolist())

        # Binary anomaly label: any timestep with label > 0
        binary_label = int((window_labels > 0).any())

        # Phase label: majority vote
        counts = np.bincount(window_labels.astype(np.int64), minlength=4)
        phase_label = int(counts.argmax())

        # Curriculum difficulty
        if len(unique_phases) == 1:
            difficulty = DIFFICULTY.get(phase_label, 0.0)
        else:
            difficulty = 0.5  # mixed-phase window

        windows.append({
            "x": x.astype(np.float32),
            "y_target": y_target.astype(np.float32),
            "label": binary_label,
            "label_seq": window_labels.astype(np.int64),   # (W,) per-timestep labels
            "phase_label": phase_label,
            "difficulty": difficulty,
        })

    return windows


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BatchDistillationDataset(Dataset):
    """
    Dataset of sliding windows from a list of batch distillation experiments.

    Parameters
    ----------
    experiments : list of (data, labels, meta)
        Output of loader.load_all_experiments() filtered to the desired split.
    normalise : bool
        Apply per-experiment normalisation (default True).
    augment_fn : callable, optional
        Function applied to x tensor during __getitem__ (training augmentation).
    difficulty_threshold : float, optional
        If set, only include windows with difficulty <= threshold (curriculum).
    tab_feature_cache : dict, optional
        (op, name) → np.ndarray (D_TAB,) precomputed tabular features.
    text_embedding_cache : dict, optional
        (op, name) → np.ndarray (384,) precomputed SBERT embeddings for operation
        logs. If None, x_text is filled with zeros (text encoder will see fallback).
    mol_composition_cache : dict, optional
        (op, name) → np.ndarray (n_mol,) mean molar fractions from GC data.
        Used as composition weights for the molecular GCN embeddings.
        If None, equal weights are used (uniform composition).
    audio_feature_cache : dict, optional
        (op, name) → np.ndarray (1, n_mels, T_frames) per-experiment log-mel
        spectrogram summary (see src/data/audio.py). If None, x_audio is
        filled with zeros (audio encoder sees fallback, matching tab/text).
    nmr_feature_cache : dict, optional
        (op, name) → np.ndarray (N_NMR_FEATURES,) per-experiment NMR
        composition summary (see src/data/nmr.py). If None, x_nmr is
        filled with zeros (Section 5.9 A15 extension).
    image_feature_cache : dict, optional
        (op, name) → np.ndarray (N_IMAGE_FEATURES,) per-experiment frozen
        ResNet-18 camera-frame summary (see src/data/image_cache.py). If
        None, x_image is filled with zeros (Section 5.9 A14 extension).
    """

    SBERT_DIM = 384   # Sentence-BERT embedding dimension
    N_MELS = 64
    AUDIO_FRAMES = 128
    N_NMR_FEATURES = 4
    N_IMAGE_FEATURES = 512

    def __init__(
        self,
        experiments: List[Tuple[np.ndarray, np.ndarray, ExperimentMeta]],
        normalise: bool = True,
        augment_fn=None,
        difficulty_threshold: Optional[float] = None,
        tab_feature_cache: Optional[Dict] = None,    # (op, name) → np.ndarray (D_TAB,)
        text_embedding_cache: Optional[Dict] = None,  # (op, name) → np.ndarray (384,)
        mol_composition_cache: Optional[Dict] = None, # (op, name) → np.ndarray (n_mol,)
        audio_feature_cache: Optional[Dict] = None,   # (op, name) → np.ndarray (1,n_mels,T)
        nmr_feature_cache: Optional[Dict] = None,     # (op, name) → np.ndarray (N_NMR_FEATURES,)
        image_feature_cache: Optional[Dict] = None,   # (op, name) → np.ndarray (N_IMAGE_FEATURES,)
    ):
        self.augment_fn = augment_fn
        self.windows: List[Dict] = []
        self._d_tab: int = 0
        self._n_mol: int = 0

        for data, labels, meta in experiments:
            if normalise:
                data = normalise_experiment(data)

            # Retrieve static tabular features for this experiment
            tab_feat: Optional[np.ndarray] = None
            if tab_feature_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                tab_feat = tab_feature_cache.get(key)
            if tab_feat is not None:
                self._d_tab = len(tab_feat)

            # Retrieve precomputed SBERT text embedding for this experiment
            text_emb: Optional[np.ndarray] = None
            if text_embedding_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                text_emb = text_embedding_cache.get(key)

            # Retrieve GC molar fraction composition for this experiment
            mol_comp: Optional[np.ndarray] = None
            if mol_composition_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                mol_comp = mol_composition_cache.get(key)
            if mol_comp is not None:
                self._n_mol = len(mol_comp)

            # Retrieve per-experiment log-mel audio summary
            audio_feat: Optional[np.ndarray] = None
            if audio_feature_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                audio_feat = audio_feature_cache.get(key)

            # Retrieve per-experiment NMR composition summary (Section 5.9 A15)
            nmr_feat: Optional[np.ndarray] = None
            if nmr_feature_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                nmr_feat = nmr_feature_cache.get(key)

            # Retrieve per-experiment ResNet-18 image summary (Section 5.9 A14)
            image_feat: Optional[np.ndarray] = None
            if image_feature_cache is not None:
                key = (meta.operating_point, meta.experiment_name)
                image_feat = image_feature_cache.get(key)

            exp_windows = build_windows(data, labels)
            for w in exp_windows:
                w["operating_point"] = meta.operating_point
                w["experiment"] = meta.experiment_name
                w["tab"]           = tab_feat   # may be None
                w["x_text"]        = text_emb  # may be None (→ zeros fallback)
                w["mol_composition"] = mol_comp # may be None (→ equal-weight fallback)
                w["x_audio"]        = audio_feat  # may be None (→ zeros fallback)
                w["x_nmr"]          = nmr_feat    # may be None (→ zeros fallback)
                w["x_image"]        = image_feat  # may be None (→ zeros fallback)
            self.windows.extend(exp_windows)

        # Apply curriculum filter
        if difficulty_threshold is not None:
            self.windows = [w for w in self.windows if w["difficulty"] <= difficulty_threshold]

        self._labels = np.array([w["label"] for w in self.windows], dtype=np.int64)
        self._difficulties = np.array([w["difficulty"] for w in self.windows], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        x = torch.from_numpy(w["x"])          # (W, V)
        y = torch.from_numpy(w["y_target"])   # (H, N_CONT)

        if self.augment_fn is not None:
            x = self.augment_fn(x)

        item = {
            "x": x,
            "y_target": y,
            "label": torch.tensor(w["label"], dtype=torch.long),
            "label_seq": torch.from_numpy(w["label_seq"]),   # (W,) per-timestep
            "phase_label": torch.tensor(w["phase_label"], dtype=torch.long),
            "difficulty": torch.tensor(w["difficulty"], dtype=torch.float32),
        }

        # Tabular features (fallback to zeros if missing)
        if w.get("tab") is not None:
            item["tab"] = torch.from_numpy(w["tab"])
        else:
            item["tab"] = torch.zeros(max(1, self._d_tab), dtype=torch.float32)

        # Text (SBERT) embedding (fallback to zeros if not precomputed)
        if w.get("x_text") is not None:
            item["x_text"] = torch.from_numpy(w["x_text"].astype(np.float32))
        else:
            item["x_text"] = torch.zeros(self.SBERT_DIM, dtype=torch.float32)

        # Molecular GC composition (molar fractions) — fallback to equal weights
        if w.get("mol_composition") is not None:
            item["mol_composition"] = torch.from_numpy(
                w["mol_composition"].astype(np.float32)
            )
        elif self._n_mol > 0:
            item["mol_composition"] = torch.full(
                (self._n_mol,), 1.0 / self._n_mol, dtype=torch.float32
            )

        # Audio (log-mel) summary — fallback to zeros if not precomputed
        if w.get("x_audio") is not None:
            item["x_audio"] = torch.from_numpy(w["x_audio"].astype(np.float32))
        else:
            item["x_audio"] = torch.zeros(
                1, self.N_MELS, self.AUDIO_FRAMES, dtype=torch.float32
            )

        # NMR composition summary (Section 5.9 A15) — fallback to zeros
        if w.get("x_nmr") is not None:
            item["x_nmr"] = torch.from_numpy(w["x_nmr"].astype(np.float32))
        else:
            item["x_nmr"] = torch.zeros(self.N_NMR_FEATURES, dtype=torch.float32)

        # Precomputed ResNet-18 image summary (Section 5.9 A14) — fallback to zeros
        if w.get("x_image") is not None:
            item["x_image"] = torch.from_numpy(w["x_image"].astype(np.float32))
        else:
            item["x_image"] = torch.zeros(self.N_IMAGE_FEATURES, dtype=torch.float32)

        # String metadata for experiment-level evaluation (returned as strings, not tensors)
        item["experiment"]      = w.get("experiment", "unknown")
        item["operating_point"] = w.get("operating_point", "unknown")

        return item

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def difficulty_scores(self) -> np.ndarray:
        return self._difficulties

    def anomaly_rate(self) -> float:
        return float(self._labels.mean())

    def summary(self) -> str:
        n = len(self.windows)
        n_anom = int(self._labels.sum())
        return (
            f"BatchDistillationDataset: {n} windows  "
            f"({n_anom} anomalous, {n - n_anom} normal, "
            f"{100 * n_anom / max(n, 1):.1f}% anomaly rate)"
        )
