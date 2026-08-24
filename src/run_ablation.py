"""
Ablation study runner: trains all 11 configurations (A1–A11) and generates result tables.

Usage:
    python src/run_ablation.py [--train-epochs 80] [--batch-size 64] [--device cuda]
                              [--output ablation_results.json]

Generates:
    - ablation_results.json: full results dictionary
    - ablation_table.txt: human-readable results table with AUPRC focus
    - ablation_auprc_only.txt: AUPRC comparison only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data.loader import load_all_experiments
from src.data.splits import find_leak_free_split
from src.data.dataset import (
    BatchDistillationDataset,
    N_INPUT_VARS,
    N_CONTINUOUS,
    WINDOW_SIZE,
)
from src.data.tabular import build_static_feature_cache, D_TAB
from src.data.molecular import build_molecule_graphs, build_mol_composition_cache
from src.data.audio import build_audio_feature_cache
from src.data.text_embed import build_text_embedding_cache
from src.data.augment import TrainingAugmentation
from src.models.utopya import UTOPYAModel, count_parameters
from src.training.loss import UTOPYALoss
from src.training.train import pretrain_tcn, train_utopya
from src.evaluation.evaluate import evaluate_utopya, evaluate_baselines, print_comparison
from src.ablations import ABLATIONS, list_ablations


DATA_ROOT = config.DATA_ROOT
SYSTEM = config.SYSTEM


def parse_args():
    p = argparse.ArgumentParser(
        description="Run multimodal ablation study (A1–A11)"
    )
    p.add_argument("--train-epochs", type=int, default=80)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default=config.CHECKPOINTS_ROOT)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", type=str,
                   default=os.path.join(config.PACKAGE_ROOT, "ablation_results.json"))
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--ablations", type=str, default="A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12",
                   help="Comma-separated list of ablations to run (default: all)")
    return p.parse_args()


def run_ablation(
    ablation_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    n_epochs: int,
    n_pretrain_epochs: int,
    batch_size: int,
    lr: float,
    save_dir: str,
) -> dict:
    """Train and evaluate a single ablation configuration. Returns metrics dict."""
    config = ABLATIONS[ablation_name]
    print(f"\n{'='*70}")
    print(f"  {ablation_name}: {config.description}")
    print(f"{'='*70}")

    os.makedirs(save_dir, exist_ok=True)
    ablation_save_dir = os.path.join(save_dir, ablation_name)
    os.makedirs(ablation_save_dir, exist_ok=True)

    # Pre-train TCN (shared for all ablations; skip if already done)
    tcn_weights_path = os.path.join(ablation_save_dir, "tcn_pretrained.pt")
    if not os.path.exists(tcn_weights_path):
        print(f"\n[1/4] Pre-training TCN...")
        pretrain_model = pretrain_tcn(
            train_loader,
            n_vars=N_INPUT_VARS,
            dmodel=128,
            n_epochs=n_pretrain_epochs,
            lr=lr,
            device=device,
            save_path=tcn_weights_path,
        )
    else:
        print(f"\n[1/4] Loading cached TCN weights from {tcn_weights_path}")
        pretrain_model = None

    # Build model with ablation config
    print(f"\n[2/4] Building {ablation_name} model...")
    model = UTOPYAModel(
        d_tab=D_TAB,
        n_vars=N_INPUT_VARS,
        n_cont=N_CONTINUOUS,
        horizon=60,
        window=WINDOW_SIZE,
        dmodel=128,
        use_tabular=config.use_tabular,
        use_text=config.use_text,
        use_gc=config.use_gc,
        use_audio=config.use_audio,
        use_nmr=config.use_nmr,
        use_image=config.use_image,
        dropout=0.5,
    ).to(device)

    # Register molecule graphs if GC modality is active (Bug 2 fix)
    if config.use_gc:
        from src.data.molecular import build_molecule_graphs
        mol_graphs = build_molecule_graphs(SYSTEM, device=torch.device(device))
        if mol_graphs is not None:
            model.register_molecule_graphs(mol_graphs)
            print(f"  Registered {len(mol_graphs)} molecule graphs in model.")
        else:
            print("  WARNING: Could not build molecule graphs — GC modality will use zeros.")

    print(f"  Parameters: {count_parameters(model):,}")

    # Load pre-trained TCN weights if available
    if os.path.exists(tcn_weights_path):
        model.tcn.load_state_dict(
            torch.load(tcn_weights_path, map_location=device)
        )
        print(f"  Loaded pretrained TCN weights.")

    # Physics loss
    from src.data.dataset import CONTINUOUS_COL_INDICES
    from src.data.loader import ALL_COLS

    MONO_COLS = ["T703", "T709", "T711", "T712"]
    mono_local = [
        CONTINUOUS_COL_INDICES.index(ALL_COLS.index(c))
        for c in MONO_COLS
        if c in ALL_COLS and ALL_COLS.index(c) in CONTINUOUS_COL_INDICES
    ]
    loss_fn = UTOPYALoss(mono_col_indices=mono_local)

    # Train
    print(f"\n[3/4] Training {ablation_name}...")
    history = train_utopya(
        model=model,
        train_dataset=train_loader.dataset,
        val_dataset=val_loader.dataset,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-3,
        device=device,
        save_dir=ablation_save_dir,
        curriculum_warmup=20,
        zero_tabular=config.zero_tabular,
        zero_text=config.zero_text,
        zero_gc=config.zero_gc,
    )

    # Load best checkpoint
    best_ckpt = os.path.join(ablation_save_dir, "utopya_best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, val={ckpt['val_loss']:.4f})")

    # Evaluate
    print(f"\n[4/4] Evaluating {ablation_name}...")
    utopya_metrics = evaluate_utopya(
        model, test_loader, device=device, split=f"{ablation_name} [test]",
        zero_tabular=config.zero_tabular,
        zero_text=config.zero_text,
        zero_gc=config.zero_gc,
    )

    return utopya_metrics


def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Ablations to run: {args.ablations}")

    # Parse which ablations to run
    ablation_names = [a.strip() for a in args.ablations.split(",")]
    for name in ablation_names:
        if name not in ABLATIONS:
            print(f"ERROR: Unknown ablation {name}")
            sys.exit(1)

    # Load data once
    print("\n[Setup] Loading data...")
    t0 = time.time()
    all_exps = load_all_experiments(DATA_ROOT)
    metas = [m for _, _, m in all_exps]
    print(f"  Loaded {len(all_exps)} experiments in {time.time() - t0:.1f}s")

    # Build tabular feature cache
    print("\n[Setup] Building tabular feature cache...")
    exp_meta_list = [{"op": m.operating_point, "name": m.experiment_name} for m in metas]
    tab_cache = build_static_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)

    # Build molecular composition cache (GC molar fractions per experiment)
    print("\n[Setup] Building molecular composition cache...")
    mol_comp_cache = build_mol_composition_cache(DATA_ROOT, SYSTEM, exp_meta_list)

    # Build audio log-mel cache and text SBERT cache (for A3/A5/A11 audio and
    # A4/A6/A7/A8/A9/A10/A11 text configs — previously always zero-filled
    # because these caches were never built here; see ADAPTATIONS.md).
    print("\n[Setup] Building audio feature cache...")
    audio_cache = build_audio_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    print("\n[Setup] Building text embedding cache...")
    text_cache = build_text_embedding_cache(DATA_ROOT, SYSTEM, exp_meta_list)

    # Create splits
    print("\n[Setup] Creating leak-free splits...")
    train_idx, val_idx, test_idx = find_leak_free_split(metas)
    train_exps = [all_exps[i] for i in train_idx]
    val_exps = [all_exps[i] for i in val_idx]
    test_exps = [all_exps[i] for i in test_idx]

    train_ds = BatchDistillationDataset(
        train_exps, normalise=True,
        augment_fn=TrainingAugmentation(),
        tab_feature_cache=tab_cache,
        mol_composition_cache=mol_comp_cache,
        audio_feature_cache=audio_cache,
        text_embedding_cache=text_cache,
    )
    val_ds = BatchDistillationDataset(
        val_exps, normalise=True,
        tab_feature_cache=tab_cache,
        mol_composition_cache=mol_comp_cache,
        audio_feature_cache=audio_cache,
        text_embedding_cache=text_cache,
    )
    test_ds = BatchDistillationDataset(
        test_exps, normalise=True,
        tab_feature_cache=tab_cache,
        mol_composition_cache=mol_comp_cache,
        audio_feature_cache=audio_cache,
        text_embedding_cache=text_cache,
    )
    print(f"  {train_ds.summary()}")
    print(f"  Val:  {len(val_ds)} windows")
    print(f"  Test: {len(test_ds)} windows")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Run ablations
    results = {}
    for ablation_name in ablation_names:
        try:
            metrics = run_ablation(
                ablation_name,
                train_loader,
                val_loader,
                test_loader,
                device=args.device,
                n_epochs=args.train_epochs,
                n_pretrain_epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_dir=args.save_dir,
            )
            results[ablation_name] = {
                k: v for k, v in metrics.items() if k != "pred_mae_per_var"
            }
        except Exception as e:
            print(f"ERROR in {ablation_name}: {e}")
            import traceback

            traceback.print_exc()
            results[ablation_name] = {"error": str(e)}

    # Save results
    print(f"\n{'='*70}")
    print(f"  SAVING RESULTS")
    print(f"{'='*70}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {args.output}")

    # Generate human-readable tables
    generate_result_tables(results, args.output)


def generate_result_tables(results: dict, output_json: str):
    """Generate human-readable result tables from ablation results."""
    output_dir = os.path.dirname(output_json) or "."

    # Table 1: Full metrics with AUPRC focus
    table_path = os.path.join(output_dir, "ablation_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("MULTIMODAL ABLATION STUDY (Table 4 equivalent) - AUPRC Focus\n")
        f.write("=" * 90 + "\n")
        f.write(
            f"{'Config':<8} {'Description':<35} "
            f"{'AUPRC':<10} {'Window AUROC':<12} {'Exp AUROC':<12} "
            f"{'Multi-signal':<12}\n"
        )
        f.write("-" * 90 + "\n")

        for ablation_name in list_ablations():
            if ablation_name not in results:
                continue
            metrics = results[ablation_name]
            if "error" in metrics:
                f.write(f"{ablation_name:<8} {'ERROR':<35} {str(metrics['error'])}\n")
                continue

            config = ABLATIONS[ablation_name]
            auprc = metrics.get("window_auprc", float("nan"))
            auroc = metrics.get("window_auroc", float("nan"))
            exp_auroc = metrics.get("experiment_auroc", float("nan"))
            multi_sig = metrics.get("multisignal_auroc", float("nan"))

            auprc_str = f"{auprc:.4f}" if not np.isnan(auprc) else "NA"
            auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "NA"
            exp_str = f"{exp_auroc:.4f}" if not np.isnan(exp_auroc) else "NA"
            multi_str = f"{multi_sig:.4f}" if not np.isnan(multi_sig) else "NA"

            f.write(
                f"{ablation_name:<8} {config.description:<35} "
                f"{auprc_str:<10} {auroc_str:<12} {exp_str:<12} {multi_str:<12}\n"
            )

        f.write("=" * 90 + "\n")
        f.write("Higher is better for all metrics\n\n")

    print(f"  Saved table to {table_path}")

    # Table 2: AUPRC only for quick comparison
    auprc_path = os.path.join(output_dir, "ablation_auprc_only.txt")
    with open(auprc_path, "w", encoding="utf-8") as f:
        f.write("WINDOW-LEVEL AUPRC COMPARISON (Paper Table 4, AUPRC column)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Config':<8} {'Description':<35} {'AUPRC':<12}\n")
        f.write("-" * 60 + "\n")

        for ablation_name in list_ablations():
            if ablation_name not in results:
                continue
            metrics = results[ablation_name]
            if "error" in metrics:
                continue

            config = ABLATIONS[ablation_name]
            auprc = metrics.get("window_auprc", float("nan"))
            auprc_str = f"{auprc:.4f}" if not np.isnan(auprc) else "NA"

            is_best = (
                ablation_name == "A7"
            )  # A7 is typically best per paper
            marker = " <- Full multimodal" if is_best else ""
            f.write(
                f"{ablation_name:<8} {config.description:<35} {auprc_str:<12}{marker}\n"
            )

        f.write("=" * 60 + "\n")

    print(f"  Saved AUPRC table to {auprc_path}")


if __name__ == "__main__":
    main()
