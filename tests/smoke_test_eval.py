"""
Evaluation smoke test — verifies metrics and baseline pipeline on the Zenodo
dataset (Arweiler et al. 2026). Does NOT run full training; uses random model
weights to check shapes/logic.

Run from project root:
    python -m tests.smoke_test_eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import config

DEVICE = config.DEVICE
print(f"Device: {DEVICE}")

# -------------------------------------------------------------------------
# Load data (reuse existing pipeline)
# -------------------------------------------------------------------------
print("\n[1/4] Loading data...")
from src.data.loader import load_all_experiments
from src.data.splits import find_leak_free_split
from src.data.dataset import BatchDistillationDataset, N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE
from src.data.tabular import build_static_feature_cache, D_TAB

all_exps  = load_all_experiments(config.DATA_ROOT, verbose=False)
metas     = [m for _, _, m in all_exps]
tab_cache = build_static_feature_cache(
    config.DATA_ROOT,
    config.SYSTEM,
    [{"op": m.operating_point, "name": m.experiment_name} for m in metas],
)

train_idx, val_idx, test_idx = find_leak_free_split(metas)
# Use only a subset for speed
train_exps = [all_exps[i] for i in train_idx[:10]]
test_exps  = [all_exps[i] for i in test_idx[:5]]

train_ds = BatchDistillationDataset(train_exps, normalise=True, tab_feature_cache=tab_cache)
test_ds  = BatchDistillationDataset(test_exps,  normalise=True, tab_feature_cache=tab_cache)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)
print(f"  Train: {len(train_ds)} windows   Test: {len(test_ds)} windows")

# -------------------------------------------------------------------------
# 2. UTOPYA evaluation (untrained model — just checks metric shapes)
# -------------------------------------------------------------------------
print("\n[2/4] UTOPYA evaluation (random weights)...")
from src.models.utopya import UTOPYAModel
from src.evaluation.evaluate import evaluate_utopya

model = UTOPYAModel(
    d_tab=D_TAB, n_vars=N_INPUT_VARS, n_cont=N_CONTINUOUS,
    use_text=False, use_mol=False,
).to(DEVICE)

utopya_metrics = evaluate_utopya(model, test_loader, device=DEVICE, split="test")

# -------------------------------------------------------------------------
# 3. Baselines (small / fast config for smoke test)
# -------------------------------------------------------------------------
print("\n[3/4] Baseline evaluation (fast config)...")
from src.evaluation.baselines import (
    PCABaseline, FFAutoencoderBaseline, IsolationForestBaseline, LSTMAutoencoderBaseline
)
from src.evaluation.evaluate import _make_window_arrays, _baseline_metrics

X_train, _,      _    = _make_window_arrays(train_loader)
X_test,  y_test, ids_ = _make_window_arrays(test_loader)
print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

baseline_results = {}

pca = PCABaseline(n_components=20).fit(X_train)
scores = pca.score(X_test)
baseline_results["PCA"] = _baseline_metrics("PCA", scores, y_test, ids_)

ffae = FFAutoencoderBaseline(latent_dim=32, n_epochs=3, device=DEVICE).fit(X_train)
scores = ffae.score(X_test)
baseline_results["FF-AE"] = _baseline_metrics("FF-AE", scores, y_test, ids_)

iso = IsolationForestBaseline(n_estimators=50, contamination=0.15).fit(X_train)
scores = iso.score(X_test)
baseline_results["IsoForest"] = _baseline_metrics("IsoForest", scores, y_test, ids_)

lstm_ae = LSTMAutoencoderBaseline(hidden=16, n_layers=1, n_epochs=3, device=DEVICE).fit(X_train)
scores = lstm_ae.score(X_test)
baseline_results["LSTM-AE"] = _baseline_metrics("LSTM-AE", scores, y_test, ids_)

# -------------------------------------------------------------------------
# 4. Comparison table
# -------------------------------------------------------------------------
print("\n[4/4] Comparison table...")
from src.evaluation.evaluate import print_comparison
print_comparison(utopya_metrics, baseline_results)

print("Evaluation smoke test passed!")
