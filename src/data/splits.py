"""
Leak-free train / validation / test split for the ternary system.

Paper spec (Section 4.1):
- Focus on ternary butan-1-ol + propan-2-ol + water (91 experiments)
- No operating point appears in more than one partition
- 55 train (14 normal, 41 anomalous, 19 operating points)
- 16 val   ( 6 normal, 10 anomalous,  8 operating points)
- 20 test  ( 8 normal, 12 anomalous,  8 operating points)
- Selected via random search over 5000 seeds optimising the above criteria

This module reproduces the search and caches the chosen seed / assignment.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from src.data.loader import ExperimentMeta


def _compute_split_score(
    train_metas: List[ExperimentMeta],
    val_metas: List[ExperimentMeta],
    test_metas: List[ExperimentMeta],
) -> float:
    """
    Return a scalar score for a candidate split (higher = better).

    Criteria:
      1. Zero operating-point overlap between any two partitions (hard constraint;
         returns -inf if violated).
      2. Close to target normal/anomalous ratios.
      3. Sufficient normal experiments in all partitions.
    """
    def _ops(metas):
        return {m.operating_point for m in metas}

    # Hard: no operating-point overlap
    if _ops(train_metas) & _ops(val_metas):
        return float("-inf")
    if _ops(train_metas) & _ops(test_metas):
        return float("-inf")
    if _ops(val_metas) & _ops(test_metas):
        return float("-inf")

    def _counts(metas):
        n_normal = sum(1 for m in metas if not m.is_anomalous)
        n_anom = sum(1 for m in metas if m.is_anomalous)
        return n_normal, n_anom

    # Targets
    tn, ta = 14, 41
    vn, va = 6, 10
    en, ea = 8, 12

    tr_n, tr_a = _counts(train_metas)
    v_n, v_a = _counts(val_metas)
    e_n, e_a = _counts(test_metas)

    # Penalise deviations from targets
    score = 0.0
    score -= abs(tr_n - tn) * 2 + abs(tr_a - ta)
    score -= abs(v_n - vn) * 2 + abs(v_a - va)
    score -= abs(e_n - en) * 2 + abs(e_a - ea)

    # Penalise if any partition has zero normal experiments
    if tr_n == 0 or v_n == 0 or e_n == 0:
        score -= 100

    return score


def find_leak_free_split(
    all_metas: List[ExperimentMeta],
    n_seeds: int = 5000,
    train_frac: float = 55 / 91,
    val_frac: float = 16 / 91,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Search over random seeds to find a leak-free operating-point split.

    Returns indices (into all_metas) for train, val, test.
    """
    # Group experiment indices by operating point
    op_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(all_metas):
        op_to_indices[m.operating_point].append(i)

    ops = list(op_to_indices.keys())
    n_total = len(all_metas)

    # Target EXPERIMENT counts per partition, derived from the fractions
    # (NOT applied to the operating-point count -- operating points don't
    # have a uniform number of experiments each).
    target_train = round(n_total * train_frac)
    target_val = round(n_total * val_frac)
    target_test = n_total - target_train - target_val

    best_score = float("-inf")
    best_split = None

    for trial_idx in range(n_seeds):
        # Fresh, independent seed per trial (not a continued shuffle stream).
        rng = random.Random(seed + trial_idx)
        shuffled = ops[:]
        rng.shuffle(shuffled)

        # Greedy bin-packing: walk operating points one at a time in shuffled
        # order and assign each one (with all of its experiments) to whichever
        # partition currently has the largest shortfall relative to its
        # target experiment count. This adapts to the uneven number of
        # experiments per operating point, unlike a fixed op-count cut.
        counts = {"train": 0, "val": 0, "test": 0}
        targets = {"train": target_train, "val": target_val, "test": target_test}
        op_buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        for op in shuffled:
            shortfalls = {name: targets[name] - counts[name] for name in counts}
            best_name = max(shortfalls, key=lambda name: shortfalls[name])
            op_buckets[best_name].append(op)
            counts[best_name] += len(op_to_indices[op])

        train_idx = [i for op in op_buckets["train"] for i in op_to_indices[op]]
        val_idx = [i for op in op_buckets["val"] for i in op_to_indices[op]]
        test_idx = [i for op in op_buckets["test"] for i in op_to_indices[op]]

        train_m = [all_metas[i] for i in train_idx]
        val_m = [all_metas[i] for i in val_idx]
        test_m = [all_metas[i] for i in test_idx]

        score = _compute_split_score(train_m, val_m, test_m)
        if score > best_score:
            best_score = score
            best_split = (sorted(train_idx), sorted(val_idx), sorted(test_idx))

    assert best_split is not None
    train_idx, val_idx, test_idx = best_split

    # Print summary
    def _summary(name, idx):
        metas = [all_metas[i] for i in idx]
        n_normal = sum(1 for m in metas if not m.is_anomalous)
        n_anom = sum(1 for m in metas if m.is_anomalous)
        ops = len({m.operating_point for m in metas})
        print(f"  {name}: {len(idx):3d} experiments  "
              f"({n_normal} normal, {n_anom} anomalous, {ops} operating points)")

    print(f"Best split (score={best_score:.1f}):")
    _summary("Train", train_idx)
    _summary("Val  ", val_idx)
    _summary("Test ", test_idx)

    return train_idx, val_idx, test_idx
