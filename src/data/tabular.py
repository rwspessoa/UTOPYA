"""
Static tabular feature extractor for UTOPYA.

Extracts a fixed-length numerical vector from per-experiment plant info and
ambient conditions CSVs. This vector is the input to TabularEncoder.

Numerical settings extracted (10 features):
    operating_point_pressure, operating_point_reflux_ratio_set,
    operating_point_heating_power_H701, operating_point_heating_power_H002,
    operating_point_cooling_temperature,
    setpoint_controller_H702..H708, setpoint_controller_PD702

Ambient conditions (4 features):
    Laboratory atmospheric pressure, Laboratory temperature,
    Laboratory humidity, Outside temperature

Total d_tab = 14  (missing values → 0, normalised by known typical ranges).
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Property names as they appear in settings.csv (col " Property" with leading space)
SETTINGS_NUMERICAL = [
    "operating_point_pressure",
    "operating_point_reflux_ratio_set",
    "operating_point_heating_power_H701",
    "operating_point_heating_power_H002",
    "operating_point_cooling_temperature",
    "setpoint_controller_H702",
    "setpoint_controller_H704",
    "setpoint_controller_H706",
    "setpoint_controller_H708",
    "setpoint_controller_PD702",
]

# Ambient Names as they appear in ambient CSV (col "Name")
AMBIENT_NUMERICAL = [
    "Laboratory atmospheric pressure",
    "Laboratory temperature",
    "Laboratory humidity",
    "Outside temperature",
]

D_TAB = len(SETTINGS_NUMERICAL) + len(AMBIENT_NUMERICAL)   # 14

# Typical scale ranges for normalisation (rough physical ranges)
# (min, max) tuples aligned with SETTINGS_NUMERICAL + AMBIENT_NUMERICAL
_SCALES = [
    (100, 1000),    # pressure mbar
    (0.5, 5),       # reflux ratio
    (0, 600),       # H701 W
    (0, 600),       # H002 W
    (-10, 50),      # cooling temp °C
    (20, 120),      # H702 °C
    (20, 120),      # H704 °C
    (20, 120),      # H706 °C
    (20, 120),      # H708 °C
    (0, 20),        # PD702 mbar
    (980, 1030),    # lab pressure mbar
    (15, 35),       # lab temp °C
    (20, 90),       # humidity %
    (-10, 40),      # outside temp °C
]


def _parse_numeric(val) -> Optional[float]:
    """Extract first float from a potentially messy string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def extract_static_features(
    plant_info_dir: str,    # e.g. .../06_.../op1/test_anormal_experiment_001
    ambient_csv:    str,    # e.g. .../04_.../op1/test_anormal_experiment_001.csv
) -> np.ndarray:
    """
    Returns a float32 numpy array of shape (D_TAB,) in [0, 1] (min-max scaled).
    Missing values are imputed with 0.5 (midrange).
    """
    raw = []

    # -- settings.csv --
    settings_path = os.path.join(plant_info_dir, "settings.csv")
    if os.path.exists(settings_path):
        df = pd.read_csv(settings_path)
        df.columns = [c.strip() for c in df.columns]
        df["Property"] = df["Property"].str.strip()
        prop_to_val = dict(zip(df["Property"], df["Value"]))
        for prop in SETTINGS_NUMERICAL:
            raw.append(_parse_numeric(prop_to_val.get(prop)))
    else:
        raw.extend([None] * len(SETTINGS_NUMERICAL))

    # -- ambient CSV --
    if os.path.exists(ambient_csv):
        df = pd.read_csv(ambient_csv)
        df.columns = [c.strip() for c in df.columns]
        if "Name" in df.columns and "Value" in df.columns:
            name_to_val = dict(zip(df["Name"].str.strip(), df["Value"]))
            for name in AMBIENT_NUMERICAL:
                raw.append(_parse_numeric(name_to_val.get(name)))
        else:
            raw.extend([None] * len(AMBIENT_NUMERICAL))
    else:
        raw.extend([None] * len(AMBIENT_NUMERICAL))

    # -- min-max normalise --
    out = []
    for v, (lo, hi) in zip(raw, _SCALES):
        if v is None:
            out.append(0.5)
        else:
            out.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))

    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pre-compute and cache for all experiments
# ---------------------------------------------------------------------------

def build_static_feature_cache(
    data_root: str,
    system:    str,
    experiments: list[dict],   # list of meta dicts with keys: op, name
) -> dict[Tuple[str, str], np.ndarray]:
    """
    Returns a dict mapping (op_name, exp_name) → feature vector of shape (D_TAB,).

    experiments: list of dicts with keys 'op' (e.g. 'operating_point_001')
                 and 'name' (e.g. 'test_anormal_experiment_001').
    """
    pi_root  = os.path.join(
        data_root,
        "06_Batch_Distillation_Plant_M-202210_Tabular_Plant_Information",
        system,
    )
    amb_root = os.path.join(
        data_root,
        "04_Batch_Distillation_Plant_M-202210_Tabular_Ambient_Conditions",
        system,
    )

    cache = {}
    for meta in experiments:
        op   = meta["op"]
        name = meta["name"]
        pi_dir   = os.path.join(pi_root, op, name)
        amb_file = os.path.join(amb_root, op, f"{name}.csv")
        feat = extract_static_features(pi_dir, amb_file)
        cache[(op, name)] = feat

    return cache
