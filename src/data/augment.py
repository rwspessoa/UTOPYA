"""
Training-time data augmentation for UTOPYA main training (Section 4.2).

Distinct from src/models/pretrain.py's augment_view() (which builds the two
contrastive views for TCN self-supervised pretraining and previously used a
fixed jitter sigma). This module is applied to the SUPERVISED training
dataset via BatchDistillationDataset's `augment_fn` hook — which existed as
a parameter since the dataset was first written but was never actually
wired into src/run.py or src/run_ablation.py, so no augmentation was ever
applied to any reported training run (see ADAPTATIONS.md / PROCEDURES_AUDIT.md).

Three complementary strategies (paper Section 4.2):
  (i)   jitter       : additive Gaussian noise, sigma ~ Uniform(0.01, 0.05), p=0.5
  (ii)  scaling      : multiplicative factor ~ Uniform(0.9, 1.1), p=0.5
  (iii) time warping : smooth nonlinear time distortion, 4 knots, max warp 0.1, p=0.3
"""

from __future__ import annotations

import random

import numpy as np
import torch


def _jitter(x: torch.Tensor, sigma_range=(0.01, 0.05)) -> torch.Tensor:
    sigma = random.uniform(*sigma_range)
    return x + torch.randn_like(x) * sigma


def _scaling(x: torch.Tensor, lo: float = 0.9, hi: float = 1.1) -> torch.Tensor:
    # x: (W, V) — one scale factor per variable (broadcast over time)
    scale = torch.empty(1, x.shape[-1]).uniform_(lo, hi)
    return x * scale


def _time_warp(x: torch.Tensor, n_knots: int = 4, max_warp: float = 0.1) -> torch.Tensor:
    """
    Smooth nonlinear time distortion: perturb `n_knots` interior control
    points of an otherwise-identity time-index mapping by up to `max_warp`
    fraction of the window length, spline-smooth via cubic interpolation of
    the (knot_time, knot_offset) pairs, then resample x along the warped
    time axis via linear interpolation.

    x : (W, V)
    """
    W = x.shape[0]
    # Control points: fixed endpoints at 0 and W-1, `n_knots` interior knots
    knot_t = np.linspace(0, W - 1, n_knots + 2)
    max_offset = max_warp * W
    knot_offset = np.zeros(n_knots + 2)
    knot_offset[1:-1] = np.random.uniform(-max_offset, max_offset, size=n_knots)

    warped_knot_t = np.clip(knot_t + knot_offset, 0, W - 1)
    # Enforce monotonicity (a valid time warp cannot go backwards)
    warped_knot_t = np.maximum.accumulate(warped_knot_t)

    # Smooth mapping from original time index -> warped time index
    t_orig = np.arange(W)
    warped_t = np.interp(t_orig, knot_t, warped_knot_t)

    x_np = x.numpy()
    out = np.empty_like(x_np)
    for v in range(x_np.shape[1]):
        out[:, v] = np.interp(warped_t, t_orig, x_np[:, v])
    return torch.from_numpy(out).to(x.dtype)


class TrainingAugmentation:
    """
    Callable matching BatchDistillationDataset's `augment_fn(x) -> x` hook.

    x : (W, V) input window tensor.
    Applies jitter (p=0.5), scaling (p=0.5), time-warp (p=0.3) independently,
    each optional so a window may receive any combination (including none).
    """

    def __init__(
        self,
        jitter_p: float = 0.5,
        jitter_sigma_range=(0.01, 0.05),
        scaling_p: float = 0.5,
        scaling_range=(0.9, 1.1),
        warp_p: float = 0.3,
        warp_knots: int = 4,
        warp_max: float = 0.1,
    ):
        self.jitter_p = jitter_p
        self.jitter_sigma_range = jitter_sigma_range
        self.scaling_p = scaling_p
        self.scaling_range = scaling_range
        self.warp_p = warp_p
        self.warp_knots = warp_knots
        self.warp_max = warp_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.jitter_p:
            x = _jitter(x, self.jitter_sigma_range)
        if random.random() < self.scaling_p:
            x = _scaling(x, *self.scaling_range)
        if random.random() < self.warp_p:
            x = _time_warp(x, self.warp_knots, self.warp_max)
        return x
