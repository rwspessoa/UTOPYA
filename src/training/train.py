"""
Full training loop for UTOPYA (Section 4.2).

Training schedule:
  Phase 1 (pretrain):  TCN self-supervised pretraining (block masking + NT-Xent)
  Phase 2 (finetune):  Full model multi-task training with curriculum learning

Curriculum learning (Section 4.2, Table 3):
  Easy→Hard sample ordering using difficulty scores from BatchDistillationDataset.
  Difficulty score per window: 0.0=normal, 0.3=anomalous, 0.6=recovery, 0.9=blind, 0.5=mixed.
  Epoch weight w(e) = min(1, e / E_warmup) used to gradually introduce hard samples.

Optimiser: AdamW, lr=3e-4, weight_decay=1e-3  (paper Section 4.2)
Scheduler: cosine annealing with linear warmup (T_max=E_total)
Grad clip: max_norm=1.0
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.pretrain import TCNPretrainer
from src.models.utopya import UTOPYAModel
from src.training.loss import UTOPYALoss
from src.data.dataset import BatchDistillationDataset, CONTINUOUS_COL_INDICES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_curriculum_sampler(
    dataset: BatchDistillationDataset,
    epoch:   int,
    E_warmup: int = 20,
) -> WeightedRandomSampler:
    """
    Weight each window so that hard samples are drawn more frequently as
    training progresses (curriculum schedule, Eq. in Section 4.2).
    """
    difficulty = dataset.difficulty_scores   # (N,) float
    alpha = min(1.0, epoch / max(1, E_warmup))
    weights  = 1.0 + alpha * difficulty      # easy→1, hard→1+alpha*score
    sampler  = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def _warmup_lr(
    optimizer: AdamW, step: int, warmup_steps: int, base_lr: float,
    group_scales: Optional[List[float]] = None,
):
    """Linear warmup, respecting each param group's relative LR scale (e.g.
    the TCN group's tcn_lr_scale) instead of collapsing all groups to the
    same rate during the warmup ramp."""
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        for i, pg in enumerate(optimizer.param_groups):
            scale = group_scales[i] if group_scales is not None else 1.0
            pg["lr"] = base_lr * scale * frac


# ---------------------------------------------------------------------------
# Pre-training phase
# ---------------------------------------------------------------------------

def pretrain_tcn(
    pretrain_loader: DataLoader,
    n_vars:   int = 31,
    dmodel:   int = 128,
    n_epochs: int = 20,
    lr:       float = 3e-4,
    device:   str = "cuda",
    save_path: Optional[str] = None,
) -> TCNPretrainer:
    """Train TCN with self-supervised objectives; return trained pretrainer."""
    from src.models.tcn import TCNEncoder
    pretrain_model = TCNPretrainer(n_input_vars=n_vars, dmodel=dmodel).to(device)
    optimiser      = AdamW(pretrain_model.parameters(), lr=lr, weight_decay=1e-3)

    print(f"[Pretrain] Start — {sum(p.numel() for p in pretrain_model.parameters()):,} params")

    for epoch in range(1, n_epochs + 1):
        pretrain_model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in pretrain_loader:
            x = batch["x"].to(device)   # (B, T, N)
            optimiser.zero_grad()
            loss, l_r, l_c = pretrain_model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()

        elapsed = time.time() - t0
        mean_loss = total_loss / max(1, len(pretrain_loader))
        print(f"  [epoch {epoch:3d}/{n_epochs}] loss={mean_loss:.4f}  ({elapsed:.1f}s)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(pretrain_model.encoder.state_dict(), save_path)
        print(f"[Pretrain] TCN weights saved -> {save_path}")

    return pretrain_model


# ---------------------------------------------------------------------------
# Fine-tuning phase
# ---------------------------------------------------------------------------

def train_utopya(
    model:          UTOPYAModel,
    train_dataset:  BatchDistillationDataset,
    val_dataset:    BatchDistillationDataset,
    loss_fn:        UTOPYALoss,
    n_epochs:       int = 80,
    batch_size:     int = 64,
    lr:             float = 3e-4,
    weight_decay:   float = 1e-3,
    warmup_epochs:  int = 5,
    curriculum_warmup: int = 20,
    device:         str = "cuda",
    save_dir:       str = "checkpoints",
    freeze_tcn_epochs: int = 3,
    tcn_lr_scale:   float = 0.1,
    modality_dropout_p: float = 0.2,
    curriculum:     bool = True,
    # Ablation zero-flags (Bug 6): passed on every forward call
    zero_tabular:   bool = False,
    zero_text:      bool = False,
    zero_gc:        bool = False,
) -> Dict[str, List[float]]:
    """
    Main UTOPYA training loop with curriculum learning.

    The TCN encoder is frozen for the first `freeze_tcn_epochs` epochs (paper:
    3) to allow the static encoders and attention layers to stabilise before
    fine-tuning the pretrained TCN; once unfrozen it trains at `tcn_lr_scale`
    (paper: 0.1, i.e. 10x smaller) of the base learning rate rather than the
    full rate used by the rest of the model (Section 4.2, "gradual
    unfreezing").

    Independent per-modality dropout (Section 4.2): each non-essential
    modality (tabular, text, GC, audio, NMR, image) is independently zeroed
    with probability `modality_dropout_p` on every TRAINING batch (not at
    validation/eval time), regardless of any fixed ablation zero-flag —
    time-series is never dropped, matching the paper.

    Returns dict of loss histories (for plotting).
    """
    os.makedirs(save_dir, exist_ok=True)

    tcn_param_ids = {id(p) for p in model.tcn.parameters()}
    tcn_params    = [p for p in model.parameters() if id(p) in tcn_param_ids]
    other_params  = [p for p in model.parameters() if id(p) not in tcn_param_ids]
    param_groups  = [
        {"params": other_params, "lr": lr},
        {"params": tcn_params,   "lr": lr * tcn_lr_scale},
    ]
    loss_params = list(loss_fn.parameters())   # non-empty only if learnable_weights=True
    if loss_params:
        param_groups.append({"params": loss_params, "lr": lr})
    optimiser = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs)

    history: Dict[str, List[float]] = {
        k: [] for k in ("train_total", "train_pred", "train_cls", "val_total")
    }

    warmup_steps_total = warmup_epochs * max(1, len(train_dataset) // batch_size)
    global_step = 0
    best_val = float("inf")

    # Continuous column indices within the full 31-var array → within N_cont
    # For labels: binary classification (label > 0 → anomaly)
    n_vars = model.loc_head.net[-1].out_features   # infer N_vars

    print(f"[Train] Start — {sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(1, n_epochs + 1):

        # --- Gradual unfreezing of TCN ---
        for p in model.tcn.parameters():
            p.requires_grad = (epoch > freeze_tcn_epochs)

        # --- Build curriculum DataLoader ---
        if curriculum:
            sampler = _make_curriculum_sampler(train_dataset, epoch, curriculum_warmup)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
        else:
            # Ablation: plain uniform sampling, ignoring difficulty scores entirely.
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # === Training ===
        model.train()
        ep_losses: Dict[str, float] = {k: 0.0 for k in ("total", "pred", "cls")}
        n_batches = 0

        for batch in train_loader:
            x_ts   = batch["x"].to(device)           # (B, T, N)
            y_true = batch["y_target"].to(device)     # (B, H, N_cont)
            labels = batch["label_seq"].to(device)    # (B, W) per-timestep
            x_tab  = batch["tab"].to(device)          # (B, d_tab)
            x_text = batch.get("x_text")
            if x_text is not None:
                x_text = x_text.to(device)            # (B, 384) SBERT embedding
            mol_composition = batch.get("mol_composition")
            if mol_composition is not None:
                mol_composition = mol_composition.to(device)   # (B, n_mol)
            x_audio = batch.get("x_audio")
            if x_audio is not None:
                x_audio = x_audio.to(device)          # (B, 1, n_mels, T_audio)
            x_nmr = batch.get("x_nmr")
            if x_nmr is not None:
                x_nmr = x_nmr.to(device)              # (B, N_NMR_FEATURES)
            x_image = batch.get("x_image")
            if x_image is not None:
                x_image = x_image.to(device)          # (B, N_IMAGE_FEATURES)

            # window-level binary label: 1 if any timestep is anomalous
            cls_label = (labels.max(dim=1).values > 0).long()   # (B,)

            # per-timestep per-variable localisation label
            # shape (B, T, N_vars) — broadcast label across variables
            loc_label = (labels > 0).float().unsqueeze(-1).expand(
                -1, -1, n_vars
            )   # (B, T, N_vars)

            # Independent per-modality stochastic dropout (training only):
            # each non-essential modality is additionally zeroed this batch
            # with probability modality_dropout_p, on top of any fixed
            # ablation zero-flag (OR'd together — either source disables it).
            batch_zero_tabular = zero_tabular or (random.random() < modality_dropout_p)
            batch_zero_text    = zero_text    or (random.random() < modality_dropout_p)
            batch_zero_gc      = zero_gc      or (random.random() < modality_dropout_p)
            batch_zero_audio   = random.random() < modality_dropout_p
            batch_zero_nmr     = random.random() < modality_dropout_p
            batch_zero_image   = random.random() < modality_dropout_p

            # Forward
            group_scales = [1.0, tcn_lr_scale] + ([1.0] if loss_params else [])
            _warmup_lr(optimiser, global_step, warmup_steps_total, lr, group_scales)
            optimiser.zero_grad()

            y_hat, cls_logits, loc_logits = model(
                x_ts, x_tab,
                x_text=x_text,
                mol_composition=mol_composition,
                x_audio=x_audio,
                x_nmr=x_nmr,
                x_image=x_image,
                zero_tabular=batch_zero_tabular,
                zero_text=batch_zero_text,
                zero_gc=batch_zero_gc,
                zero_audio=batch_zero_audio,
                zero_nmr=batch_zero_nmr,
                zero_image=batch_zero_image,
            )

            losses = loss_fn(y_hat, y_true, cls_logits, cls_label, loc_logits, loc_label)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            for k in ("total", "pred", "cls"):
                ep_losses[k] += losses[k].item()

            global_step += 1
            n_batches   += 1

        scheduler.step()

        for k in ("total", "pred", "cls"):
            ep_losses[k] /= max(1, n_batches)
        history["train_total"].append(ep_losses["total"])
        history["train_pred"].append(ep_losses["pred"])
        history["train_cls"].append(ep_losses["cls"])

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_ts   = batch["x"].to(device)
                y_true = batch["y_target"].to(device)
                labels = batch["label_seq"].to(device)   # (B, W) per-timestep
                x_tab  = batch["tab"].to(device)
                x_text = batch.get("x_text")
                if x_text is not None:
                    x_text = x_text.to(device)
                mol_composition = batch.get("mol_composition")
                if mol_composition is not None:
                    mol_composition = mol_composition.to(device)
                x_audio = batch.get("x_audio")
                if x_audio is not None:
                    x_audio = x_audio.to(device)
                x_nmr = batch.get("x_nmr")
                if x_nmr is not None:
                    x_nmr = x_nmr.to(device)
                x_image = batch.get("x_image")
                if x_image is not None:
                    x_image = x_image.to(device)

                cls_label = (labels.max(dim=1).values > 0).long()
                loc_label = (labels > 0).float().unsqueeze(-1).expand(-1, -1, n_vars)

                # No stochastic modality dropout at validation time — only
                # fixed ablation zero-flags apply, matching normal eval.
                y_hat, cls_logits, loc_logits = model(
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
                losses = loss_fn(y_hat, y_true, cls_logits, cls_label, loc_logits, loc_label)
                val_loss += losses["total"].item()

        val_loss /= max(1, len(val_loader))
        history["val_total"].append(val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(save_dir, "utopya_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_loss":             val_loss,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            frozen = "TCN frozen" if epoch <= freeze_tcn_epochs else "TCN unfrozen"
            print(
                f"  [epoch {epoch:3d}/{n_epochs}] "
                f"train={ep_losses['total']:.4f}  "
                f"pred={ep_losses['pred']:.4f}  "
                f"cls={ep_losses['cls']:.4f}  "
                f"val={val_loss:.4f}  [{frozen}]"
            )

    print(f"[Train] Done. Best val loss: {best_val:.4f}")
    return history
