"""
Standalone reconstruction-head anomaly scoring (paper Section 5.7.3).

Reconstructs the paper's claim: "we trained the reconstruction head
separately on normal windows only, with the encoder and fusion layers
frozen... reconstruction errors achieve a standalone test AUROC of 0.695...
the best three-signal combination (classification, reconstruction error,
and prediction error) reaches 0.855." Neither the standalone reconstruction
score nor the three-signal combination had any corresponding code in this
repo before (see PROCEDURES_AUDIT.md) — the ReconstructionHead module
itself was added to src/models/fusion.py/utopya.py as part of this pass.

Steps:
  1. Load the (procedure-fixed) A7 checkpoint, freeze everything except
     recon_head, train recon_head only on NORMAL-label training windows.
  2. Score val+test (ALL windows) by reconstruction MSE -> standalone AUROC.
  3. Combine with existing cls_prob + pred_error signals via the paper-
     faithful experiment-level rank fusion (src/evaluation/metrics.py) to
     get a three-signal AUROC, searching the two fusion weights on val.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from src.data.loader import load_all_experiments
from src.data.splits import find_leak_free_split
from src.data.dataset import BatchDistillationDataset, N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE
from src.data.tabular import build_static_feature_cache, D_TAB
from src.data.molecular import build_molecule_graphs, build_mol_composition_cache
from src.data.audio import build_audio_feature_cache
from src.data.text_embed import build_text_embedding_cache
from src.models.utopya import UTOPYAModel
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA_ROOT = config.DATA_ROOT
SYSTEM = config.SYSTEM
DEVICE = config.DEVICE
OUT_DIR = os.path.join(config.CHECKPOINTS_ROOT, "A7_recon_head")
find_a7_checkpoint = config.find_a7_checkpoint

_ap = argparse.ArgumentParser()
_ap.add_argument("--epochs", type=int, default=15)
N_EPOCHS = _ap.parse_args().epochs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1/4] Loading data + caches...")
    all_exps = load_all_experiments(DATA_ROOT, verbose=False)
    metas = [m for _, _, m in all_exps]
    exp_meta_list = [{"op": m.operating_point, "name": m.experiment_name} for m in metas]

    tab_cache = build_static_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    mol_cache = build_mol_composition_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    audio_cache = build_audio_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    text_cache = build_text_embedding_cache(DATA_ROOT, SYSTEM, exp_meta_list)

    train_idx, val_idx, test_idx = find_leak_free_split(metas)
    train_exps = [all_exps[i] for i in train_idx]
    val_exps = [all_exps[i] for i in val_idx]
    test_exps = [all_exps[i] for i in test_idx]

    def make_ds(exps):
        return BatchDistillationDataset(
            exps, normalise=True,
            tab_feature_cache=tab_cache, mol_composition_cache=mol_cache,
            audio_feature_cache=audio_cache, text_embedding_cache=text_cache,
        )

    train_ds_full, val_ds, test_ds = make_ds(train_exps), make_ds(val_exps), make_ds(test_exps)
    normal_idx = [i for i, lbl in enumerate(train_ds_full.labels) if lbl == 0]
    train_ds_normal = Subset(train_ds_full, normal_idx)
    print(f"  Train (normal-only): {len(train_ds_normal)}/{len(train_ds_full)} windows  "
          f"Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds_normal, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    print("[2/4] Loading A7 backbone, freezing all but recon_head...")
    model = UTOPYAModel(
        d_tab=D_TAB, n_vars=N_INPUT_VARS, n_cont=N_CONTINUOUS,
        horizon=60, window=WINDOW_SIZE, dmodel=128,
        use_tabular=True, use_text=True, use_gc=True, use_audio=True,
        dropout=0.0,
    ).to(DEVICE)
    graphs = build_molecule_graphs(SYSTEM, device=torch.device(DEVICE))
    if graphs is not None:
        model.register_molecule_graphs(graphs)
    ckpt_path = find_a7_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded backbone from {ckpt_path}")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.recon_head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.recon_head.parameters(), lr=1e-3, weight_decay=1e-4)

    def _forward_batch(batch):
        x_ts = batch["x"].to(DEVICE)
        x_tab = batch["tab"].to(DEVICE)
        x_text = batch.get("x_text")
        x_text = x_text.to(DEVICE) if x_text is not None else None
        mol_comp = batch.get("mol_composition")
        mol_comp = mol_comp.to(DEVICE) if mol_comp is not None else None
        x_audio = batch.get("x_audio")
        x_audio = x_audio.to(DEVICE) if x_audio is not None else None
        _, _, _, recon_hat = model(
            x_ts, x_tab, x_text=x_text, mol_composition=mol_comp, x_audio=x_audio,
            return_reconstruction=True,
        )
        return x_ts, recon_hat

    print(f"[3/4] Training recon_head for {N_EPOCHS} epochs on normal-only windows...")
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, N_EPOCHS + 1):
        model.eval()
        model.recon_head.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            x_ts, recon_hat = _forward_batch(batch)
            loss = F.mse_loss(recon_hat, x_ts)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_ts, recon_hat = _forward_batch(batch)
                val_loss += F.mse_loss(recon_hat, x_ts).item()
                n_val += 1
        val_loss /= max(1, n_val)
        print(f"  [epoch {epoch:2d}/{N_EPOCHS}] train_loss={total_loss/max(1,n_batches):.4f} "
              f"val_recon_mse={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.recon_head.state_dict().items()}

    model.recon_head.load_state_dict(best_state)
    print(f"  Best val recon MSE: {best_val_loss:.4f}")

    print("[4/4] Scoring val+test by reconstruction error, computing standalone + "
          "three-signal AUROC...")

    def _collect(loader):
        model.eval()
        recon_errs, cls_probs, pred_errs, labels, exp_ids = [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                x_ts = batch["x"].to(DEVICE)
                y_true = batch["y_target"].to(DEVICE)
                x_tab = batch["tab"].to(DEVICE)
                x_text = batch.get("x_text")
                x_text = x_text.to(DEVICE) if x_text is not None else None
                mol_comp = batch.get("mol_composition")
                mol_comp = mol_comp.to(DEVICE) if mol_comp is not None else None
                x_audio = batch.get("x_audio")
                x_audio = x_audio.to(DEVICE) if x_audio is not None else None
                lab = batch["label_seq"]

                y_hat, cls_logits, _, recon_hat = model(
                    x_ts, x_tab, x_text=x_text, mol_composition=mol_comp, x_audio=x_audio,
                    return_reconstruction=True,
                )
                recon_err = F.mse_loss(recon_hat, x_ts, reduction="none").mean(dim=(1, 2))
                pred_mae = F.l1_loss(y_hat, y_true, reduction="none").mean(dim=(1, 2))
                cls_prob = F.softmax(cls_logits, dim=-1)[:, 1]
                win_label = (lab.max(dim=1).values > 0).long()

                recon_errs.append(recon_err.cpu().numpy())
                pred_errs.append(pred_mae.cpu().numpy())
                cls_probs.append(cls_prob.cpu().numpy())
                labels.append(win_label.numpy())
                exp_ids.extend(f"{op}|{exp}" for op, exp in
                                zip(batch["operating_point"], batch["experiment"]))
        return (np.concatenate(recon_errs), np.concatenate(cls_probs),
                np.concatenate(pred_errs), np.concatenate(labels), exp_ids)

    val_recon, val_cls, val_pred, val_lab, val_ids = _collect(val_loader)
    test_recon, test_cls, test_pred, test_lab, test_ids = _collect(test_loader)

    # Standalone reconstruction-error AUROC (window-level, matches how the
    # paper reports the standalone reconstruction AUROC of 0.695).
    standalone_val_auroc = roc_auc_score(val_lab, val_recon)
    standalone_test_auroc = roc_auc_score(test_lab, test_recon)
    print(f"  Standalone reconstruction AUROC: val={standalone_val_auroc:.4f} "
          f"test={standalone_test_auroc:.4f}")

    # Three-signal experiment-level rank fusion: search (w_cls, w_recon) with
    # w_pred = 1 - w_cls - w_recon on a coarse grid, select by val AUROC.
    def _exp_aggregate(cls, recon, pred, lab, ids):
        from collections import defaultdict
        agg_cls, agg_recon, agg_pred, agg_lab = defaultdict(list), defaultdict(list), defaultdict(list), {}
        for c, r, p, l, eid in zip(cls, recon, pred, lab, ids):
            agg_cls[eid].append(c)
            agg_recon[eid].append(r)
            agg_pred[eid].append(p)
            agg_lab[eid] = max(agg_lab.get(eid, 0), int(l))
        eids = sorted(agg_cls.keys())
        max_cls = np.array([max(agg_cls[e]) for e in eids])
        p95_recon = np.array([np.percentile(agg_recon[e], 95) for e in eids])
        p95_pred = np.array([np.percentile(agg_pred[e], 95) for e in eids])
        lab_arr = np.array([agg_lab[e] for e in eids])
        return max_cls, p95_recon, p95_pred, lab_arr

    val_mc, val_mr, val_mp, val_el = _exp_aggregate(val_cls, val_recon, val_pred, val_lab, val_ids)
    test_mc, test_mr, test_mp, test_el = _exp_aggregate(test_cls, test_recon, test_pred, test_lab, test_ids)

    def _fused_auroc(max_cls, p95_recon, p95_pred, labels, w_cls, w_recon):
        w_pred = 1.0 - w_cls - w_recon
        r_cls = rankdata(max_cls) / len(max_cls)
        r_recon = rankdata(p95_recon) / len(p95_recon)
        r_pred = rankdata(p95_pred) / len(p95_pred)
        fused = w_cls * r_cls + w_recon * r_recon + w_pred * r_pred
        if len(np.unique(labels)) < 2:
            return float("nan")
        return roc_auc_score(labels, fused)

    best = (0.71, 0.10, -1.0)   # paper's own reported weights as a starting default
    for w_cls in np.linspace(0, 1, 11):
        for w_recon in np.linspace(0, 1 - w_cls, 11):
            auc = _fused_auroc(val_mc, val_mr, val_mp, val_el, w_cls, w_recon)
            if not np.isnan(auc) and auc > best[2]:
                best = (w_cls, w_recon, auc)

    w_cls_opt, w_recon_opt, val_three_signal_auroc = best
    test_three_signal_auroc = _fused_auroc(test_mc, test_mr, test_mp, test_el, w_cls_opt, w_recon_opt)
    test_two_signal_auroc = roc_auc_score(test_el, 0.5 * (rankdata(test_mc) / len(test_mc)) +
                                           0.5 * (rankdata(test_mp) / len(test_mp)))
    print(f"  Optimal weights (val-selected): w_cls={w_cls_opt:.2f} w_recon={w_recon_opt:.2f} "
          f"w_pred={1-w_cls_opt-w_recon_opt:.2f}")
    print(f"  Three-signal experiment AUROC: val={val_three_signal_auroc:.4f} "
          f"test={test_three_signal_auroc:.4f}")
    print(f"  (for reference) Two-signal (cls+pred only) experiment AUROC: test={test_two_signal_auroc:.4f}")

    torch.save({"recon_head_state_dict": model.recon_head.state_dict(),
                "val_recon_mse": best_val_loss}, os.path.join(OUT_DIR, "recon_head.pt"))
    results = {
        "a7_checkpoint_used": ckpt_path,
        "standalone_reconstruction_auroc_val": float(standalone_val_auroc),
        "standalone_reconstruction_auroc_test": float(standalone_test_auroc),
        "three_signal_weight_cls": float(w_cls_opt),
        "three_signal_weight_recon": float(w_recon_opt),
        "three_signal_weight_pred": float(1 - w_cls_opt - w_recon_opt),
        "three_signal_auroc_val": float(val_three_signal_auroc),
        "three_signal_auroc_test": float(test_three_signal_auroc),
        "two_signal_auroc_test_for_reference": float(test_two_signal_auroc),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved recon_head checkpoint + metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
