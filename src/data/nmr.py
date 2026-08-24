"""
NMR composition cache builder for UTOPYA (Section 5.9's NMR modality, A15
frozen-backbone extension).

Wires the "08_..._NMR_Composition/tabular" per-experiment composition CSVs
(mole-fraction estimates derived from in-line NMR spectra, sampled roughly
once per minute during an experiment) into a fixed-size per-experiment
summary that NMREncoder (src/models/encoders.py) can consume: the mean
mole-fraction of each chemical component plus the mean assumed-impurity
scalar, averaged over all NMR samples taken during the experiment.

Treated as an experiment-level summary (constant across all windows of an
experiment) — the same simplification already used for Audio in this
codebase (see audio.py's own docstring for the precedent: audio, GC
composition and tabular static features are all applied uniformly across a
window rather than varying per-window).

Different chemical systems in this dataset have different numbers of
components (e.g. a binary system would only have 2 mole-fraction columns,
not 3), so the mole-fraction column names are kept in a small per-system
registry, NMR_MOL_FRACTION_COLS, mirroring src/data/molecular.py's
GC_MOL_FRACTION_COLS. Only the ternary system is populated since that is the
only system this codebase's training pipeline uses (see src/run.py's SYSTEM
constant), but the registry is structured as a dict so it is extensible to
other systems later.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NMR_ROOT_DIRNAME = "08_Batch_Distillation_Plant_M-202210_NMR_Composition"

# Mole-fraction CSV column names per chemical system (order defines feature order).
NMR_MOL_FRACTION_COLS: Dict[str, List[str]] = {
    "batch_dist_ternary_butan-1-ol+propan-2-ol+water": [
        "x_butan-1-ol",
        "x_propan-2-ol",
        "x_water",
    ],
}

# Feature width for the ternary system: 3 mole fractions + 1 impurity scalar.
N_NMR_FEATURES = 4


def _load_experiment_nmr(
    data_root: str,
    system:    str,
    op:        str,
    name:      str,
    mol_cols:  List[str],
) -> Optional[np.ndarray]:
    """
    Read and average one experiment's NMR composition CSV.

    Returns (len(mol_cols) + 1,) float32 array of [mean mole fractions...,
    mean Assumed_Impurity], or None if the CSV does not exist.
    """
    nmr_dir  = os.path.join(data_root, NMR_ROOT_DIRNAME, "tabular", system, op)
    csv_path = os.path.join(nmr_dir, f"{name}.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    feats = []
    for col in mol_cols + ["Assumed_Impurity"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            feats.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def build_nmr_feature_cache(
    data_root:   str,
    system:      str,
    experiments: List[Dict],   # dicts with keys "op" and "name"
) -> Optional[Dict[Tuple[str, str], np.ndarray]]:
    """
    Build per-experiment NMR composition summaries.

    Returns
    -------
    dict : (op, name) → np.ndarray (len(mol_cols) + 1,) mean mole fractions
        followed by mean assumed impurity. None if the system is not in the
        registry.

    Experiments with no usable composition CSV fall back to an all-zero
    array (same convention as the audio/tabular/molecular caches).
    """
    mol_cols = NMR_MOL_FRACTION_COLS.get(system)
    if mol_cols is None:
        print(f"[NMR] System '{system}' not in registry — skipping NMR modality.")
        return None

    feature_len = len(mol_cols) + 1
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing = 0

    for meta in experiments:
        op, name = meta["op"], meta["name"]
        feat = _load_experiment_nmr(data_root, system, op, name, mol_cols)
        if feat is None:
            feat = np.zeros(feature_len, dtype=np.float32)
            missing += 1
        cache[(op, name)] = feat

    if missing:
        print(f"[NMR] {missing}/{len(experiments)} experiments had no usable "
              f"composition CSV — zero-filled fallback.")
    print(f"[NMR] Built composition cache for {len(cache)} experiments "
          f"({feature_len} features).")
    return cache
