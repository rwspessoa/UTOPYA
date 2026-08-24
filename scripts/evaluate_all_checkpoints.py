"""
Evaluate every available checkpoint (A1-A12 ablations + B0-B3/C0-C3 grid
runs) on BOTH the validation and test splits, to reconstruct the paper's
val_vs_test_scatter.png and test_auroc_progression.png (which need val
AUROC per config — our ablation_results*.json files are test-only).
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from common import load_model_and_data
from src.data.dataset import BatchDistillationDataset, N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE
from src.data.tabular import D_TAB
from src.data.molecular import build_molecule_graphs
from src.models.utopya import UTOPYAModel
from src.ablations import ABLATIONS
from src.evaluation.evaluate import evaluate_utopya

OUT = os.path.join(config.OUTPUTS_ROOT, "collected")
os.makedirs(OUT, exist_ok=True)

ctx = load_model_and_data(load_checkpoint=False)  # only data/caches needed here
device = ctx["device"]

val_exps = [ctx["all_exps"][i] for i in ctx["val_idx"]]
test_exps = [ctx["all_exps"][i] for i in ctx["test_idx"]]


def make_ds(exps):
    return BatchDistillationDataset(
        exps, normalise=True,
        tab_feature_cache=ctx["tab_cache"],
        mol_composition_cache=ctx["mol_cache"],
        audio_feature_cache=ctx["audio_cache"],
        text_embedding_cache=ctx["text_cache"],
    )


val_ds, test_ds = make_ds(val_exps), make_ds(test_exps)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)


def build_model_for_ablation(name):
    a = ABLATIONS[name]
    m = UTOPYAModel(
        d_tab=D_TAB, n_vars=N_INPUT_VARS, n_cont=N_CONTINUOUS,
        horizon=60, window=WINDOW_SIZE, dmodel=128,
        use_tabular=a.use_tabular, use_text=a.use_text,
        use_gc=a.use_gc, use_audio=a.use_audio, dropout=0.0,
    ).to(device)
    if a.use_gc:
        graphs = build_molecule_graphs(config.SYSTEM, device=torch.device(device))
        if graphs is not None:
            m.register_molecule_graphs(graphs)
    return m


def build_model_a7():
    return build_model_for_ablation("A7")


rows = []

# --- A1..A12 ablation checkpoints ---
for name in sorted(ABLATIONS.keys(), key=lambda n: int(n[1:])):
    ckpt_path = os.path.join(config.CHECKPOINTS_ROOT, name, "utopya_best.pt")
    if not os.path.exists(ckpt_path):
        continue
    try:
        model = build_model_for_ablation(name)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        val_m = evaluate_utopya(model, val_loader, device=device, split=f"{name}-val")
        test_m = evaluate_utopya(model, test_loader, device=device, split=f"{name}-test")
        rows.append({"config": name, "group": "modality_ablation",
                     "val_auroc": val_m["window_auroc"], "test_auroc": test_m["window_auroc"]})
        print(f"{name}: val={val_m['window_auroc']:.4f} test={test_m['window_auroc']:.4f}")
    except Exception as e:
        print(f"SKIP {name}: {e}")

# --- B0-B3 / C0-C3 grid checkpoints (all built on A7 architecture) ---
grid_glob = os.path.join(config.OUTPUTS_ROOT, "*", "checkpoints", "utopya_best.pt")
for run_dir in sorted(glob.glob(grid_glob)):
    run_id = run_dir.split(os.sep)[-3]
    try:
        model = build_model_a7()
        ckpt = torch.load(run_dir, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        val_m = evaluate_utopya(model, val_loader, device=device, split=f"{run_id}-val")
        test_m = evaluate_utopya(model, test_loader, device=device, split=f"{run_id}-test")
        group = "physics_grid" if run_id.startswith("phys_") else (
            "curriculum_grid" if run_id.startswith("curr_") else "sensitivity")
        rows.append({"config": run_id, "group": group,
                     "val_auroc": val_m["window_auroc"], "test_auroc": test_m["window_auroc"]})
        print(f"{run_id}: val={val_m['window_auroc']:.4f} test={test_m['window_auroc']:.4f}")
    except Exception as e:
        print(f"SKIP {run_id}: {e}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "val_test_auroc_all_configs.csv"), index=False)
print(f"\nSaved {len(df)} rows to val_test_auroc_all_configs.csv")
