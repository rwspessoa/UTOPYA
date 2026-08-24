"""
Evaluation metrics for UTOPYA (Section 4.3 and 5.1).

Metrics implemented:
  - Window-level AUROC  : ROC-AUC over all windows (primary metric)
  - Window-level AUPRC  : PR-AUC (corrects for class imbalance)
  - F1 (optimal thresh) : F1 at threshold that maximises it on val set
  - Per-variable MAE    : prediction head accuracy
  - Experiment-level AUROC (max aggregation over windows)
  - Multi-signal AUROC  : rank-based fusion of cls_prob + pred_error (Eq. Table 4 caption)

Multi-signal scoring (rank-based fusion, paper Section 5.1):
  For each window w:
    r_cls(w)  = percentile rank of P(anomaly|w) among all test windows
    r_pred(w) = percentile rank of MSE(ŷ_w, y_w)  among all test windows
    score(w)  = 0.5 * r_cls(w) + 0.5 * r_pred(w)
  Experiment-level: max score over windows in that experiment.
  AUROC computed over experiment-level labels.

Experiment-level aggregation: max window score per experiment.

NOTE (added later): the original `multisignal_auroc()` above fuses per-WINDOW
cls_prob and pred_error (MSE) ranks with a fixed 50/50 weight, then max-aggregates
to experiment level. This does not match the paper's actual procedure and is kept
ONLY for backward compatibility — it is relied on by ~70 historical result files
and must keep producing the same numbers it always has.

The paper-faithful formula (Section "Multi-Signal Experiment-Level Scoring") ranks
two EXPERIMENT-level summary statistics — max classification probability and the
95th-percentile of per-window prediction MAE, both computed per experiment — across
experiments (not across windows), then fuses those two rank arrays with a weight w
optimized on a held-out validation set (paper-reported optimum w=0.73, experiment
AUROC 0.874). This is implemented in `multisignal_auroc_paper()` and
`find_optimal_multisignal_weight()`, added below `multisignal_auroc()`. Both old and
new functions coexist: the old one for backward-compatible comparison against
previously-run configs, the new one for anything computed going forward.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Per-window score containers
# ---------------------------------------------------------------------------

class WindowResults:
    """
    Stores per-window predictions and ground-truth labels.
    Collected during an evaluation loop, then metrics computed at the end.
    """

    def __init__(self):
        self.cls_probs:    List[float] = []   # P(anomaly | window)
        self.pred_errors:  List[float] = []   # MSE(ŷ, y)
        self.labels:       List[int]   = []   # 0/1 binary anomaly label
        self.exp_ids:      List[str]   = []   # experiment identifier string
        self.pred_maes:    List[np.ndarray] = []  # per-variable MAE (N_cont,)

    def add(
        self,
        cls_prob:   float,
        pred_error: float,
        label:      int,
        exp_id:     str,
        pred_mae:   Optional[np.ndarray] = None,
    ):
        self.cls_probs.append(cls_prob)
        self.pred_errors.append(pred_error)
        self.labels.append(label)
        self.exp_ids.append(exp_id)
        if pred_mae is not None:
            self.pred_maes.append(pred_mae)

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Window-level metrics
# ---------------------------------------------------------------------------

def window_auroc(results: WindowResults) -> float:
    y_true  = np.array(results.labels)
    y_score = np.array(results.cls_probs)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def window_auprc(results: WindowResults) -> float:
    y_true  = np.array(results.labels)
    y_score = np.array(results.cls_probs)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def optimal_f1(results: WindowResults) -> Tuple[float, float]:
    """
    Find threshold that maximises F1 on this result set.
    Returns (best_f1, best_threshold).
    """
    y_true  = np.array(results.labels)
    y_score = np.array(results.cls_probs)
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_f1   = float(f1_scores[best_idx])
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_f1, best_thr


def prediction_mae(results: WindowResults) -> np.ndarray:
    """Per-variable mean absolute error across all windows."""
    if not results.pred_maes:
        return np.array([float("nan")])
    return np.stack(results.pred_maes, axis=0).mean(axis=0)


# ---------------------------------------------------------------------------
# Experiment-level aggregation (max-pool over window scores)
# ---------------------------------------------------------------------------

def _group_by_experiment(results: WindowResults, scores: np.ndarray):
    """Group per-window scores and labels by experiment, return (exp_scores, exp_labels)."""
    exp_scores: Dict[str, List[float]] = defaultdict(list)
    exp_label:  Dict[str, int]         = {}

    for score, label, exp_id in zip(scores, results.labels, results.exp_ids):
        exp_scores[exp_id].append(float(score))
        # Experiment is anomalous if ANY window is anomalous
        exp_label[exp_id] = max(exp_label.get(exp_id, 0), int(label))

    exp_ids     = sorted(exp_scores.keys())
    agg_scores  = np.array([max(exp_scores[eid]) for eid in exp_ids])
    agg_labels  = np.array([exp_label[eid]        for eid in exp_ids])
    return agg_scores, agg_labels


def experiment_auroc(results: WindowResults) -> float:
    """Experiment-level AUROC: max-aggregate window cls_prob per experiment."""
    scores = np.array(results.cls_probs)
    agg_scores, agg_labels = _group_by_experiment(results, scores)
    if len(np.unique(agg_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(agg_labels, agg_scores))


# ---------------------------------------------------------------------------
# Multi-signal rank-based fusion  (paper Table 4 caption)
# ---------------------------------------------------------------------------

def multisignal_auroc(results: WindowResults) -> float:
    """
    Rank-based fusion of cls_prob and pred_error → experiment-level AUROC.

    Steps:
      1. Percentile-rank each signal among all windows (scipy.rankdata / N)
      2. Average the two rank scores per window
      3. Aggregate to experiment level via max
      4. Compute AUROC over experiment labels
    """
    from scipy.stats import rankdata

    cls_probs   = np.array(results.cls_probs)
    pred_errors = np.array(results.pred_errors)

    # Percentile rank: higher cls_prob = more anomalous
    r_cls  = rankdata(cls_probs)  / len(cls_probs)
    # Higher pred_error = more anomalous
    r_pred = rankdata(pred_errors) / len(pred_errors)

    fused = 0.5 * r_cls + 0.5 * r_pred    # (N_windows,)

    agg_scores, agg_labels = _group_by_experiment(results, fused)
    if len(np.unique(agg_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(agg_labels, agg_scores))


# ---------------------------------------------------------------------------
# Multi-signal rank-based fusion — paper-faithful, experiment-level
# (Section "Multi-Signal Experiment-Level Scoring")
# ---------------------------------------------------------------------------

def _experiment_level_signals(
    results: WindowResults,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Group windows by experiment and compute the two paper-faithful
    experiment-level summary statistics used by multisignal_auroc_paper():

      max_cls_prob(e) = max(cls_prob over windows in experiment e)
      p95_mae(e)      = 95th percentile of per-window scalar MAE
                        (pred_maes[i].mean() over variables) over windows in e
      label(e)        = 1 if ANY window in e is anomalous else 0

    Returns (exp_ids_sorted, max_cls_prob_array, p95_mae_array, label_array),
    all aligned by the same sorted exp_ids order (mirrors the
    sorted(exp_scores.keys()) determinism pattern used in _group_by_experiment).
    """
    exp_cls_probs: Dict[str, List[float]] = defaultdict(list)
    exp_maes:      Dict[str, List[float]] = defaultdict(list)
    exp_label:     Dict[str, int]         = {}

    for i, (cls_prob, label, exp_id) in enumerate(
        zip(results.cls_probs, results.labels, results.exp_ids)
    ):
        exp_cls_probs[exp_id].append(float(cls_prob))
        exp_maes[exp_id].append(float(results.pred_maes[i].mean()))
        exp_label[exp_id] = max(exp_label.get(exp_id, 0), int(label))

    exp_ids = sorted(exp_cls_probs.keys())
    max_cls_prob = np.array([max(exp_cls_probs[eid]) for eid in exp_ids])
    p95_mae      = np.array([np.percentile(exp_maes[eid], 95) for eid in exp_ids])
    labels       = np.array([exp_label[eid] for eid in exp_ids])

    return exp_ids, max_cls_prob, p95_mae, labels


def multisignal_auroc_paper(results: WindowResults, weight: float = 0.73) -> float:
    """
    Paper-faithful multi-signal experiment-level AUROC (Section "Multi-Signal
    Experiment-Level Scoring").

    Steps:
      1. Per experiment, compute max_cls_prob and p95_mae (95th percentile of
         per-window scalar MAE) via _experiment_level_signals().
      2. Percentile-rank each signal ACROSS EXPERIMENTS (scipy.rankdata / n_exp)
         — not across windows, unlike the legacy multisignal_auroc().
      3. Fuse via weighted average: weight * r_cls + (1 - weight) * r_pred.
      4. Compute AUROC over experiment-level labels.

    `weight` defaults to the paper's reported optimum (0.73, experiment AUROC
    0.874) but callers may pass a different weight (e.g. one selected via
    find_optimal_multisignal_weight() on a validation split).
    """
    from scipy.stats import rankdata

    _, max_cls_prob, p95_mae, labels = _experiment_level_signals(results)

    if len(np.unique(labels)) < 2:
        return float("nan")

    n_exp = len(max_cls_prob)
    r_cls  = rankdata(max_cls_prob) / n_exp
    r_pred = rankdata(p95_mae)      / n_exp

    fused = weight * r_cls + (1 - weight) * r_pred

    return float(roc_auc_score(labels, fused))


def find_optimal_multisignal_weight(
    val_results: WindowResults,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Grid-search the fusion weight w in [0, 1] that maximises
    multisignal_auroc_paper() on a held-out VALIDATION WindowResults, matching
    the paper's stated protocol of optimizing w "on the validation set".

    Returns (best_weight, best_val_auroc) — the argmax weight and its
    corresponding AUROC on `val_results`. This function evaluates ONLY on the
    passed-in (validation) results; the returned best_weight is meant to then
    be applied to a separate TEST WindowResults by the caller, to avoid
    test-set leakage in the weight selection.
    """
    if grid is None:
        grid = np.linspace(0, 1, 101)

    best_weight = float("nan")
    best_auroc  = float("-inf")

    for w in grid:
        auroc = multisignal_auroc_paper(val_results, weight=float(w))
        if auroc > best_auroc:
            best_auroc  = auroc
            best_weight = float(w)

    return best_weight, best_auroc


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def compute_all_metrics(results: WindowResults, split_name: str = "test") -> Dict[str, float]:
    """Compute and return all metrics as a dict."""
    w_auroc  = window_auroc(results)
    w_auprc  = window_auprc(results)
    f1, thr  = optimal_f1(results)
    exp_auc  = experiment_auroc(results)
    ms_auc   = multisignal_auroc(results)
    mae      = prediction_mae(results)

    metrics = {
        "window_auroc":      w_auroc,
        "window_auprc":      w_auprc,
        "f1_optimal":        f1,
        "f1_threshold":      thr,
        "experiment_auroc":  exp_auc,
        "multisignal_auroc": ms_auc,
        "pred_mae_mean":     float(mae.mean()),
    }

    # Per-variable MAE (as list)
    metrics["pred_mae_per_var"] = mae.tolist()

    width = max(len(k) for k in metrics if k != "pred_mae_per_var") + 2
    print(f"\n{'='*50}")
    print(f"  Evaluation results [{split_name}]")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k == "pred_mae_per_var":
            continue
        label = k.ljust(width)
        print(f"  {label}: {v:.4f}")
    print(f"{'='*50}\n")

    return metrics
