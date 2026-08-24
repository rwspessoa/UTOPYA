"""
Main evaluation runner for UTOPYA and baselines.

Usage:
    from src.evaluation.evaluate import evaluate_utopya, evaluate_baselines, print_comparison

    results_utopya = evaluate_utopya(model, test_loader, device="cuda")
    results_all    = evaluate_baselines(train_X, train_y, test_X, test_y, test_exp_ids)
    print_comparison(results_utopya, results_all)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import WindowResults, compute_all_metrics
from src.evaluation.baselines import (
    PCABaseline,
    FFAutoencoderBaseline,
    IsolationForestBaseline,
    LSTMAutoencoderBaseline,
)
from src.data.dataset import CONTINUOUS_COL_INDICES


# ---------------------------------------------------------------------------
# UTOPYA evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_utopya(
    model,
    loader:    DataLoader,
    device:    str = "cuda",
    split:     str = "test",
    zero_tabular: bool = False,
    zero_text:    bool = False,
    zero_gc:      bool = False,
    return_raw:   bool = False,
) -> Dict:
    """
    Run model in eval mode and collect per-window predictions.

    Expects batch keys: x, y_target, label, label_seq, tab, [experiment, operating_point].

    Returns dict with all metrics.
    """
    model.eval()
    results = WindowResults()

    for batch in loader:
        x_ts   = batch["x"].to(device)          # (B, T, N)
        y_true = batch["y_target"].to(device)    # (B, H, N_cont)
        labels = batch["label_seq"]              # (B, T) on CPU
        x_tab  = batch["tab"].to(device)         # (B, d_tab)
        x_text = batch.get("x_text")
        if x_text is not None:
            x_text = x_text.to(device)           # (B, 384) SBERT embedding
        mol_composition = batch.get("mol_composition")
        if mol_composition is not None:
            mol_composition = mol_composition.to(device)   # (B, n_mol)
        x_audio = batch.get("x_audio")
        if x_audio is not None:
            x_audio = x_audio.to(device)         # (B, 1, n_mels, T_audio)
        x_nmr = batch.get("x_nmr")
        if x_nmr is not None:
            x_nmr = x_nmr.to(device)             # (B, N_NMR_FEATURES)
        x_image = batch.get("x_image")
        if x_image is not None:
            x_image = x_image.to(device)         # (B, N_IMAGE_FEATURES)

        # Build experiment IDs
        if "experiment" in batch and "operating_point" in batch:
            exp_ids = [
                f"{op}|{exp}"
                for op, exp in zip(batch["operating_point"], batch["experiment"])
            ]
        else:
            exp_ids = [str(i) for i in range(len(x_ts))]

        y_hat, cls_logits, _ = model(
            x_ts, x_tab,
            x_text=x_text,
            mol_composition=mol_composition,
            x_audio=x_audio,
            x_nmr=x_nmr,
            x_image=x_image,
            zero_tabular=zero_tabular,
            zero_text=zero_text,
            zero_gc=zero_gc,
        )

        # Classification probability P(anomaly)
        cls_probs = F.softmax(cls_logits, dim=-1)[:, 1].cpu().numpy()   # (B,)

        # Prediction MSE per window (mean over H and N_cont)
        pred_mse  = F.mse_loss(y_hat, y_true, reduction="none").mean(dim=(1, 2))
        pred_mse  = pred_mse.cpu().numpy()   # (B,)

        # Per-variable MAE
        pred_mae  = F.l1_loss(y_hat, y_true, reduction="none").mean(dim=1)  # (B, N_cont)
        pred_mae  = pred_mae.cpu().numpy()

        # Window-level binary label
        win_labels = (labels.max(dim=1).values > 0).numpy().astype(int)  # (B,)

        for b in range(len(x_ts)):
            results.add(
                cls_prob   = float(cls_probs[b]),
                pred_error = float(pred_mse[b]),
                label      = int(win_labels[b]),
                exp_id     = exp_ids[b],
                pred_mae   = pred_mae[b],
            )

    metrics = compute_all_metrics(results, split_name=f"UTOPYA [{split}]")
    if return_raw:
        return metrics, results
    return metrics


# ---------------------------------------------------------------------------
# Baselines evaluation
# ---------------------------------------------------------------------------

def _make_window_arrays(
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract (X, y_labels, exp_ids) arrays from a DataLoader."""
    Xs, ys, ids_ = [], [], []
    for batch in loader:
        Xs.append(batch["x"].numpy())
        win_labels = (batch["label_seq"].max(dim=1).values > 0).numpy().astype(int)
        ys.append(win_labels)
        if "experiment" in batch and "operating_point" in batch:
            for op, exp in zip(batch["operating_point"], batch["experiment"]):
                ids_.append(f"{op}|{exp}")
        else:
            ids_.extend([f"exp_{i}" for i in range(len(batch["x"]))])
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        ids_,
    )


def _baseline_metrics(
    name:    str,
    scores:  np.ndarray,
    labels:  np.ndarray,
    exp_ids: List[str],
) -> Dict:
    """Wrap raw scores in a WindowResults and compute metrics."""
    res = WindowResults()
    for score, label, exp_id in zip(scores, labels, exp_ids):
        res.add(
            cls_prob   = float(score),    # score used as cls_prob for AUROC
            pred_error = float(score),    # same signal (no separate pred head)
            label      = int(label),
            exp_id     = exp_id,
        )
    return compute_all_metrics(res, split_name=f"{name} [test]")


def evaluate_baselines(
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       str = "cuda",
) -> Dict[str, Dict]:
    """
    Train and evaluate all four baselines.
    Returns dict of {baseline_name: metrics_dict}.
    """
    print("Extracting window arrays from loaders...")
    X_train, _, _         = _make_window_arrays(train_loader)
    X_test,  y_test, ids_ = _make_window_arrays(test_loader)
    print(f"  train: {X_train.shape}  test: {X_test.shape}  "
          f"anomaly rate: {y_test.mean():.2%}")

    all_results = {}

    # 1. PCA
    print("\n[Baseline 1/4] PCA...")
    pca = PCABaseline().fit(X_train)
    scores = pca.score(X_test)
    all_results["PCA"] = _baseline_metrics("PCA", scores, y_test, ids_)

    # 2. Feedforward Autoencoder
    print("\n[Baseline 2/4] Feedforward Autoencoder...")
    ffae = FFAutoencoderBaseline(latent_dim=64, n_epochs=50, device=device).fit(X_train)
    scores = ffae.score(X_test)
    all_results["FF-AE"] = _baseline_metrics("FF-AE", scores, y_test, ids_)

    # 3. Isolation Forest
    print("\n[Baseline 3/4] Isolation Forest...")
    iso = IsolationForestBaseline(n_estimators=200, contamination=0.15).fit(X_train)
    scores = iso.score(X_test)
    all_results["IsoForest"] = _baseline_metrics("IsoForest", scores, y_test, ids_)

    # 4. LSTM Autoencoder
    print("\n[Baseline 4/4] LSTM Autoencoder...")
    lstm_ae = LSTMAutoencoderBaseline(
        hidden=64, n_layers=2, n_epochs=30, device=device
    ).fit(X_train)
    scores = lstm_ae.score(X_test)
    all_results["LSTM-AE"] = _baseline_metrics("LSTM-AE", scores, y_test, ids_)

    return all_results


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_comparison(
    utopya_metrics: Dict,
    baseline_metrics: Dict[str, Dict],
):
    """Print a comparison table of UTOPYA vs baselines (matches paper Table 3)."""
    all_models = {"UTOPYA": utopya_metrics, **baseline_metrics}
    cols = ["window_auroc", "window_auprc", "f1_optimal", "experiment_auroc", "multisignal_auroc"]
    col_labels = ["Win-AUROC", "Win-AUPRC", "F1-opt", "Exp-AUROC", "MS-AUROC"]

    header = f"{'Model':<14}" + "".join(f"{c:>12}" for c in col_labels)
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for model_name, metrics in all_models.items():
        row = f"{model_name:<14}"
        for col in cols:
            v = metrics.get(col, float("nan"))
            row += f"{v:>12.4f}" if not np.isnan(v) else f"{'—':>12}"
        if model_name == "UTOPYA":
            print(f"{row}  <- UTOPYA")
        else:
            delta = metrics.get("window_auroc", float("nan")) - utopya_metrics.get("window_auroc", 0.0)
            print(f"{row}  d={delta:+.4f}")

    print(sep)
