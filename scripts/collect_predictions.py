"""
Collect all per-window predictions/embeddings needed to reconstruct the
paper's figures/tables, from the A7 checkpoint (+ trained phase_head).

Writes to scripts_reconstruction/collected/:
  windows_val.csv, windows_test.csv   — one row per window (scalars)
  embeddings_val.npy, embeddings_test.npy   — (N, 128) z_fused
  mae_per_var_val.npy, mae_per_var_test.npy — (N, N_cont) per-variable AE
  baseline_scores.npz  — PCA/IsoForest/FF-AE/LSTM-AE scores on val+test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from common import load_model_and_data
from src.data.dataset import BatchDistillationDataset, CONTINUOUS_COL_INDICES
from src.data.loader import ALL_COLS

OUT = os.path.join(config.OUTPUTS_ROOT, "collected")
os.makedirs(OUT, exist_ok=True)

ctx = load_model_and_data()
model, device = ctx["model"], ctx["device"]

# Load the phase-head-trained checkpoint on top (backbone identical, +phase_head)
phase_ckpt_path = os.path.join(config.CHECKPOINTS_ROOT, "A7_v2", "utopya_best_with_phase_head.pt")
ckpt = torch.load(phase_ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print(f"Loaded phase-head checkpoint (val_acc={ckpt['phase_val_acc']:.4f})")

CONT_NAMES = [ALL_COLS[i] for i in CONTINUOUS_COL_INDICES]


def make_ds(exps):
    return BatchDistillationDataset(
        exps, normalise=True,
        tab_feature_cache=ctx["tab_cache"],
        mol_composition_cache=ctx["mol_cache"],
        audio_feature_cache=ctx["audio_cache"],
        text_embedding_cache=ctx["text_cache"],
    )


splits = {
    "train": [ctx["all_exps"][i] for i in ctx["train_idx"]],
    "val": [ctx["all_exps"][i] for i in ctx["val_idx"]],
    "test": [ctx["all_exps"][i] for i in ctx["test_idx"]],
}

collected_x = {}   # split -> (N, W, N_vars) raw window arrays for baselines

for split_name, exps in splits.items():
    ds = make_ds(exps)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    rows, embeddings, mae_per_var, xs_raw = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x_ts = batch["x"].to(device)
            x_tab = batch["tab"].to(device)
            x_text = batch.get("x_text")
            x_text = x_text.to(device) if x_text is not None else None
            x_audio = batch.get("x_audio")
            x_audio = x_audio.to(device) if x_audio is not None else None
            mol_comp = batch.get("mol_composition")
            mol_comp = mol_comp.to(device) if mol_comp is not None else None
            y_true = batch["y_target"].to(device)
            labels = batch["label_seq"]
            phase_label = batch["phase_label"]

            y_hat, cls_logits, loc_logits, z_fused, phase_logits = model(
                x_ts, x_tab, x_text=x_text, mol_composition=mol_comp, x_audio=x_audio,
                return_embedding=True, return_phase=True,
            )
            cls_probs = F.softmax(cls_logits, dim=-1)[:, 1].cpu().numpy()
            pred_mse = F.mse_loss(y_hat, y_true, reduction="none").mean(dim=(1, 2)).cpu().numpy()
            pred_mae_pv = F.l1_loss(y_hat, y_true, reduction="none").mean(dim=1).cpu().numpy()  # (B, N_cont)
            phase_pred = phase_logits.argmax(dim=-1).cpu().numpy()
            win_labels = (labels.max(dim=1).values > 0).numpy().astype(int)

            exp_ids = [f"{op}|{exp}" for op, exp in zip(batch["operating_point"], batch["experiment"])]
            for b in range(len(x_ts)):
                rows.append({
                    "split": split_name, "experiment_id": exp_ids[b],
                    "operating_point": batch["operating_point"][b],
                    "is_anomaly": int(win_labels[b]),
                    "phase_label_true": int(phase_label[b].item()),
                    "phase_label_pred": int(phase_pred[b]),
                    "cls_prob": float(cls_probs[b]),
                    "pred_mse": float(pred_mse[b]),
                    "pred_mae": float(pred_mae_pv[b].mean()),
                })
            embeddings.append(z_fused.cpu().numpy())
            mae_per_var.append(pred_mae_pv)
            xs_raw.append(x_ts.cpu().numpy())

    df = pd.DataFrame(rows)
    df["window_idx"] = range(len(df))
    df.to_csv(os.path.join(OUT, f"windows_{split_name}.csv"), index=False)
    np.save(os.path.join(OUT, f"embeddings_{split_name}.npy"), np.concatenate(embeddings, axis=0))
    np.save(os.path.join(OUT, f"mae_per_var_{split_name}.npy"), np.concatenate(mae_per_var, axis=0))
    collected_x[split_name] = np.concatenate(xs_raw, axis=0)
    print(f"{split_name}: {len(df)} windows, anomaly rate={df.is_anomaly.mean():.3f}, "
          f"phase_head acc={ (df.phase_label_true==df.phase_label_pred).mean():.4f}")

np.save(os.path.join(OUT, "cont_names.npy"), np.array(CONT_NAMES))

# ---------------------------------------------------------------------------
# Baselines: fit on train windows, score on val + test
# ---------------------------------------------------------------------------
from src.evaluation.baselines import (
    PCABaseline, FFAutoencoderBaseline, IsolationForestBaseline, LSTMAutoencoderBaseline,
)

X_train = collected_x["train"]
X_val = collected_x["val"]
X_test = collected_x["test"]
print(f"Baseline shapes: train {X_train.shape} val {X_val.shape} test {X_test.shape}")

baseline_scores = {}
print("[Baseline 1/4] PCA...")
pca = PCABaseline().fit(X_train)   # n_components=0.95 default (fixed, see PROCEDURES_AUDIT.md)
baseline_scores["PCA_val"] = pca.score(X_val)
baseline_scores["PCA_test"] = pca.score(X_test)

print("[Baseline 2/4] FF-AE...")
ffae = FFAutoencoderBaseline(latent_dim=64, n_epochs=50, device=device).fit(X_train)
baseline_scores["FF-AE_val"] = ffae.score(X_val)
baseline_scores["FF-AE_test"] = ffae.score(X_test)

print("[Baseline 3/4] IsoForest...")
iso = IsolationForestBaseline(n_estimators=200, contamination=0.15).fit(X_train)
baseline_scores["IsoForest_val"] = iso.score(X_val)
baseline_scores["IsoForest_test"] = iso.score(X_test)

print("[Baseline 4/4] LSTM-AE...")
lstm_ae = LSTMAutoencoderBaseline(hidden=64, n_layers=2, n_epochs=30, device=device).fit(X_train)
baseline_scores["LSTM-AE_val"] = lstm_ae.score(X_val)
baseline_scores["LSTM-AE_test"] = lstm_ae.score(X_test)

np.savez(os.path.join(OUT, "baseline_scores.npz"), **baseline_scores)
print("Saved baseline_scores.npz")
print("\nAll data collected in", OUT)
