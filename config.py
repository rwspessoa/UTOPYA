"""
Central configuration for the UTOPYA methodology package.

Every path used anywhere in this package is derived from here, so the
whole folder can be copied/relocated and only these values (or the
corresponding environment variables) need to change.

Environment variable overrides (all optional):
  UTOPYA_DATA_ROOT         - root of the Arweiler et al. dataset
  UTOPYA_CHECKPOINTS_ROOT  - where trained model checkpoints are saved/loaded
  UTOPYA_OUTPUTS_ROOT      - where per-run metrics.json files are written
  UTOPYA_DEVICE            - "cuda" or "cpu" (default: cuda if available)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# Root of this package (utopya_methodology/), independent of cwd.
PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_ROOT = os.environ.get(
    "UTOPYA_DATA_ROOT",
    str(PACKAGE_ROOT / "data"),
)
CHECKPOINTS_ROOT = os.environ.get(
    "UTOPYA_CHECKPOINTS_ROOT",
    str(PACKAGE_ROOT / "checkpoints"),
)
OUTPUTS_ROOT = os.environ.get(
    "UTOPYA_OUTPUTS_ROOT",
    str(PACKAGE_ROOT / "outputs"),
)
DEVICE = os.environ.get(
    "UTOPYA_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)

# The chemical system this pipeline is validated on (Arweiler et al. 2026,
# ternary butan-1-ol + propan-2-ol + water — see src/data/loader.py).
SYSTEM = "batch_dist_ternary_butan-1-ol+propan-2-ol+water"

# Preferred order for locating the headline "A7" (full multimodal) backbone
# checkpoint — first match wins. A7_v2 is the procedure-fixed retrain
# (real augmentation, 3-epoch TCN freeze + 10x LR, stochastic modality
# dropout, corrected leak-free split); A7 is the earlier bug-fixed-but-not-
# procedure-fixed run kept for comparison.
A7_CHECKPOINT_CANDIDATES = [
    os.path.join(CHECKPOINTS_ROOT, "A7_v2", "utopya_best.pt"),
    os.path.join(CHECKPOINTS_ROOT, "A7", "utopya_best.pt"),
]


def find_a7_checkpoint() -> str:
    for p in A7_CHECKPOINT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No A7 checkpoint found in {A7_CHECKPOINT_CANDIDATES}. "
        f"Train one first (see src/run.py --ablation A7)."
    )


def ensure_dirs() -> None:
    for d in (CHECKPOINTS_ROOT, OUTPUTS_ROOT):
        os.makedirs(d, exist_ok=True)
