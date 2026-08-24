"""
Low-level data loading utilities for the Arweiler et al. batch distillation dataset.

Loads per-experiment time-series (sensors + actuators) across Startup / Operation /
Shutdown phases and returns a single concatenated NumPy array with accompanying
per-timestep anomaly labels (Operation phase only; Startup and Shutdown are treated
as normal, i.e. label 0).

Label encoding (from dataset):
    0 – normal
    1 – blind   (fault started but not yet observable)
    2 – anomalous
    3 – recovery
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset path constants
# ---------------------------------------------------------------------------

SYSTEM_NAME = "batch_dist_ternary_butan-1-ol+propan-2-ol+water"
PHASES = ["Startup", "Operation", "Shutdown"]

SENSOR_COLS = [
    "LS701", "LS702", "T701", "T702", "T703", "T704", "T706", "T708",
    "T709", "T711", "T712", "T705", "FT703", "FT704", "PDI701", "PDI702",
    "PY23", "FYI702",
]
ACTUATOR_COLS = [
    "H701", "H702", "H706", "H704", "H708", "H002",
    "AV8", "AV709", "P301", "AV716", "TV1", "P701", "P702",
]
# Binary valve / pump columns that are excluded from physics constraints
# P701 and P702 are pump speed setpoints (0-100%) — continuous, not binary
BINARY_COLS = ["AV8", "AV709", "P301", "AV716", "LS701", "LS702"]

ALL_COLS = SENSOR_COLS + ACTUATOR_COLS  # 31 variables total

# Temperature sensor pairs ordered bottom→top for monotonicity constraint
# (lower_sensor, upper_sensor)
TEMP_MONOTONE_PAIRS = [
    ("T703", "T709"),
    ("T709", "T711"),
    ("T711", "T712"),
    ("T712", "T705"),
]


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Metadata for a single experiment."""
    operating_point: str          # e.g. "operating_point_001"
    experiment_name: str          # e.g. "test_anormal_experiment_001"
    is_anomalous: bool            # True if any timestep label > 0
    n_timesteps: int = 0
    label_counts: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _iter_experiments(data_root: str) -> List[Tuple[str, str]]:
    """
    Yield (operating_point, experiment_stem) pairs for the ternary system.
    Experiment stems are identified from the sensor Operation directory because
    every experiment has sensor data.
    """
    sensor_op_dir = os.path.join(
        data_root,
        "01_Batch_Distillation_Plant_M-202210_Timeseries_Sensors",
        "Operation",
        SYSTEM_NAME,
    )
    pairs = []
    for op in sorted(os.listdir(sensor_op_dir)):
        op_path = os.path.join(sensor_op_dir, op)
        if not os.path.isdir(op_path):
            continue
        for fname in sorted(os.listdir(op_path)):
            if fname.endswith(".csv"):
                pairs.append((op, fname[:-4]))  # strip .csv
    return pairs


# ---------------------------------------------------------------------------
# Single-phase CSV loading
# ---------------------------------------------------------------------------

def _load_phase_csv(
    data_root: str,
    phase: str,
    operating_point: str,
    experiment_stem: str,
) -> Optional[pd.DataFrame]:
    """Load sensor + actuator CSVs for one phase and return a merged DataFrame."""
    def _read(modality_prefix: str, cols: List[str]) -> Optional[pd.DataFrame]:
        path = os.path.join(
            data_root,
            f"{modality_prefix}_Batch_Distillation_Plant_M-202210_Timeseries_{'Sensors' if '01' in modality_prefix else 'Actuators'}",
            phase,
            SYSTEM_NAME,
            operating_point,
            f"{experiment_stem}.csv",
        )
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        available = [c for c in cols if c in df.columns]
        return df[["Time"] + available]

    sensors_df = _read("01", SENSOR_COLS)
    actuators_df = _read("02", ACTUATOR_COLS)

    if sensors_df is None:
        return None

    if actuators_df is not None:
        df = sensors_df.merge(actuators_df, on="Time", how="left")
    else:
        df = sensors_df

    # Ensure all expected columns exist (fill missing with NaN)
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[["Time"] + ALL_COLS]


def _load_label_csv(
    data_root: str,
    operating_point: str,
    experiment_stem: str,
) -> Optional[pd.Series]:
    """Load per-timestep anomaly labels for the Operation phase."""
    path = os.path.join(
        data_root,
        "00_Batch_Distillation_Plant_M-202210_Timeseries_Label_Anomaly_Metadata",
        "Operation",
        SYSTEM_NAME,
        operating_point,
        f"{experiment_stem}.csv",
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df["Label (anomaly)"].values


# ---------------------------------------------------------------------------
# Full experiment loading
# ---------------------------------------------------------------------------

def load_experiment(
    data_root: str,
    operating_point: str,
    experiment_stem: str,
) -> Tuple[np.ndarray, np.ndarray, ExperimentMeta]:
    """
    Load and concatenate Startup / Operation / Shutdown for one experiment.

    Returns
    -------
    data : np.ndarray, shape (T, 31), float32
        Concatenated time-series (all phases).
    labels : np.ndarray, shape (T,), int8
        Per-timestep anomaly labels.  Startup and Shutdown timesteps receive
        label 0 (normal); Operation timesteps use the dataset labels.
    meta : ExperimentMeta
    """
    phase_dfs = {}
    for phase in PHASES:
        df = _load_phase_csv(data_root, phase, operating_point, experiment_stem)
        if df is not None:
            phase_dfs[phase] = df

    if not phase_dfs:
        raise FileNotFoundError(
            f"No data found for {operating_point}/{experiment_stem}"
        )

    op_labels = _load_label_csv(data_root, operating_point, experiment_stem)

    # Build concatenated array and labels
    arrays, label_arrays = [], []
    for phase in PHASES:
        if phase not in phase_dfs:
            continue
        df = phase_dfs[phase]
        arr = df[ALL_COLS].values.astype(np.float32)
        arrays.append(arr)

        if phase == "Operation" and op_labels is not None:
            n = min(len(arr), len(op_labels))
            phase_labels = np.zeros(len(arr), dtype=np.int8)
            phase_labels[:n] = op_labels[:n].astype(np.int8)
        else:
            phase_labels = np.zeros(len(arr), dtype=np.int8)
        label_arrays.append(phase_labels)

    data = np.concatenate(arrays, axis=0)
    labels = np.concatenate(label_arrays, axis=0)

    # Replace NaN with 0 (missing sensors)
    np.nan_to_num(data, copy=False)

    counts = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
    meta = ExperimentMeta(
        operating_point=operating_point,
        experiment_name=experiment_stem,
        is_anomalous=bool((labels > 0).any()),
        n_timesteps=len(data),
        label_counts=counts,
    )
    return data, labels, meta


def load_all_experiments(
    data_root: str,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, ExperimentMeta]]:
    """Load all 91 experiments for the ternary system."""
    experiments = _iter_experiments(data_root)
    results = []
    for i, (op, stem) in enumerate(experiments):
        try:
            data, labels, meta = load_experiment(data_root, op, stem)
            results.append((data, labels, meta))
            if verbose and (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(experiments)} experiments")
        except Exception as exc:
            print(f"  WARNING: skipping {op}/{stem}: {exc}")
    if verbose:
        print(f"  Loaded {len(results)}/{len(experiments)} experiments total")
    return results
