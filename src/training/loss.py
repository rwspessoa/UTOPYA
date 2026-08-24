"""
Physics-informed and task losses (Section 3.6 & 4.1).

Combined loss (Eq. 9):
    L = λ_pred * L_pred  +  λ_cls * L_cls  +  λ_loc * L_loc  +  λ_phys * L_phys

L_pred  : MSE over continuous variables (prediction head)
L_cls   : Focal loss for window-level anomaly classification
L_loc   : BCE for per-timestep per-variable anomaly localisation
L_phys  : physics regularisation
    • smoothness : penalise ||Δ²y||_F  (second-order differences)
    • monotonicity: penalise positive slope on temperature columns expected
                    to be non-increasing during distillation

Temperature monotonicity pairs (from paper Table 1):
    T703, T709, T711, T712 should be non-increasing.
    T705 is controlled and excluded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Indices in CONTINUOUS_COL_INDICES that are expected to be non-increasing.
# These are resolved externally by the caller (see dataset.py for CONTINUOUS_COL_INDICES).
# Default: None → physics loss is skipped if not provided.

LAMBDA_PRED  = 1.0
LAMBDA_CLS   = 0.5
LAMBDA_LOC   = 0.5
LAMBDA_PHYS  = 0.1


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al. 2017) for imbalanced anomaly detection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    For multi-class (window-level 2-class) we use softmax + cross-entropy form.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 2) raw logits
        targets : (B,) long  0=normal, 1=anomaly
        """
        log_probs = F.log_softmax(logits, dim=-1)           # (B, 2)
        probs     = log_probs.exp()                          # (B, 2)
        target_one_hot = F.one_hot(targets, num_classes=2).float()  # (B, 2)
        p_t = (probs * target_one_hot).sum(-1)               # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce = -(log_probs * target_one_hot).sum(-1)           # (B,)
        return (self.alpha * focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# Physics regularisation
# ---------------------------------------------------------------------------

def smoothness_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Penalise large second-order differences in predicted trajectories.
    y_pred : (B, H, N_cont)
    """
    d1 = y_pred[:, 1:, :] - y_pred[:, :-1, :]      # first diffs  (B, H-1, N)
    d2 = d1[:, 1:, :] - d1[:, :-1, :]               # second diffs (B, H-2, N)
    return (d2 ** 2).mean()


def monotonicity_loss(
    y_pred: torch.Tensor, mono_col_indices: list[int]
) -> torch.Tensor:
    """
    Penalise positive slopes on temperature columns expected to be non-increasing.
    y_pred : (B, H, N_cont)
    mono_col_indices : local indices (within N_cont) of monotone-decreasing columns
    """
    if not mono_col_indices:
        return torch.tensor(0.0, device=y_pred.device)
    y_mono = y_pred[:, :, mono_col_indices]          # (B, H, K)
    diffs  = y_mono[:, 1:, :] - y_mono[:, :-1, :]   # (B, H-1, K)
    return F.relu(diffs).mean()                       # penalise increases only


class PhysicsLoss(nn.Module):
    """
    Combined physics regularisation: smoothness + monotonicity.
    """

    def __init__(
        self,
        mono_col_indices: list[int] | None = None,
        lambda_smooth: float = 1.0,
        lambda_mono:   float = 1.0,
    ):
        super().__init__()
        self.mono_col_indices = mono_col_indices or []
        self.lambda_smooth    = lambda_smooth
        self.lambda_mono      = lambda_mono

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        L_smooth = smoothness_loss(y_pred)
        L_mono   = monotonicity_loss(y_pred, self.mono_col_indices)
        return self.lambda_smooth * L_smooth + self.lambda_mono * L_mono


# ---------------------------------------------------------------------------
# Combined UTOPYA loss
# ---------------------------------------------------------------------------

class WeightedCELoss(nn.Module):
    """
    Plain class-weighted cross-entropy, used as the "remove focal loss"
    ablation arm (paper Table 5, "Remove focal loss" -> ~0.80 val AUROC,
    -0.03 val AUROC) — same class-imbalance weight w_+ as FocalLoss's
    alpha-style weighting, but without the (1-p_t)^gamma focusing term.
    """

    def __init__(self, pos_weight: float = 6.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            targets == 1,
            torch.full_like(targets, self.pos_weight, dtype=logits.dtype),
            torch.ones_like(targets, dtype=logits.dtype),
        )
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (weights * ce).mean()


class UTOPYALoss(nn.Module):
    """
    L = λ_pred*L_pred + λ_cls*L_cls + λ_loc*L_loc + λ_phys*L_phys

    All weights are configurable for ablation studies.

    use_focal : bool
        If False, replaces FocalLoss with plain weighted cross-entropy
        (WeightedCELoss) for L_cls — reconstructs the paper's "Remove focal
        loss" ablation arm (Table 5).
    learnable_weights : bool
        If True, ignores the fixed lambda_pred/lambda_cls/lambda_phys
        scalars and instead learns per-task homoscedastic-uncertainty
        weights following Kendall et al. (2018): each task loss L_i is
        combined as exp(-log_var_i) * L_i + log_var_i, with log_var_i a
        learnable nn.Parameter. Reconstructs the paper's Section 5.5 claim
        that fixed weights were compared against learned uncertainty
        weighting (this comparison did not previously exist as runnable
        code anywhere in the repo — see PROCEDURES_AUDIT.md). L_loc keeps
        its fixed lambda_loc weight either way (the paper's uncertainty-
        weighting claim is scoped to the three losses it explicitly
        names: pred, cls, phys).
    """

    def __init__(
        self,
        mono_col_indices: list[int] | None = None,
        lambda_pred:  float = LAMBDA_PRED,
        lambda_cls:   float = LAMBDA_CLS,
        lambda_loc:   float = LAMBDA_LOC,
        lambda_phys:  float = LAMBDA_PHYS,
        lambda_smooth_phys: float = 1.0,   # PhysicsLoss internal smoothness weight
        lambda_mono_phys:   float = 1.0,   # PhysicsLoss internal monotonicity weight
        use_focal: bool = True,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.lambda_pred  = lambda_pred
        self.lambda_cls   = lambda_cls
        self.lambda_loc   = lambda_loc
        self.lambda_phys  = lambda_phys
        self.use_focal    = use_focal
        self.learnable_weights = learnable_weights
        self.focal        = FocalLoss() if use_focal else WeightedCELoss()
        self.physics      = PhysicsLoss(
            mono_col_indices,
            lambda_smooth=lambda_smooth_phys,
            lambda_mono=lambda_mono_phys,
        )
        if learnable_weights:
            # log(sigma^2) per task, Kendall et al. 2018 Eq. 10/11.
            self.log_var_pred = nn.Parameter(torch.zeros(()))
            self.log_var_cls  = nn.Parameter(torch.zeros(()))
            self.log_var_phys = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        y_pred:    torch.Tensor,    # (B, H, N_cont)
        y_true:    torch.Tensor,    # (B, H, N_cont)
        cls_logit: torch.Tensor,    # (B, 2)
        cls_label: torch.Tensor,    # (B,) long 0/1
        loc_logit: torch.Tensor,    # (B, T, N_vars)
        loc_label: torch.Tensor,    # (B, T, N_vars) float 0/1
    ) -> dict[str, torch.Tensor]:

        L_pred  = F.mse_loss(y_pred, y_true)
        L_cls   = self.focal(cls_logit, cls_label)
        L_loc   = F.binary_cross_entropy_with_logits(loc_logit, loc_label)
        L_phys  = self.physics(y_pred)

        if self.learnable_weights:
            L_total = (
                torch.exp(-self.log_var_pred) * L_pred + self.log_var_pred +
                torch.exp(-self.log_var_cls)  * L_cls  + self.log_var_cls  +
                self.lambda_loc               * L_loc  +
                torch.exp(-self.log_var_phys) * L_phys + self.log_var_phys
            )
        else:
            L_total = (
                self.lambda_pred  * L_pred  +
                self.lambda_cls   * L_cls   +
                self.lambda_loc   * L_loc   +
                self.lambda_phys  * L_phys
            )
        return {
            "total": L_total,
            "pred":  L_pred,
            "cls":   L_cls,
            "loc":   L_loc,
            "phys":  L_phys,
        }
