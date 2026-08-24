"""
UTOPYA full training + evaluation run script.

Usage:
    python src/run.py [--pretrain-epochs 20] [--train-epochs 80] [--batch-size 64]
                      [--device cuda] [--save-dir checkpoints]

This script:
  1. Loads all 91 experiments
  2. Builds tabular feature cache
  3. Creates leak-free train/val/test splits
  4. Runs TCN self-supervised pretraining (Phase 1)
  5. Runs full UTOPYA fine-tuning with curriculum learning (Phase 2)
  6. Evaluates UTOPYA and four baselines on the test set
  7. Prints comparison table matching paper Table 3
"""

import argparse
import random
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.data.loader import load_all_experiments
from src.data.splits import find_leak_free_split
from src.data.dataset import (
    BatchDistillationDataset,
    N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE,
)
from src.data.tabular import build_static_feature_cache, D_TAB
from src.data.molecular import build_molecule_graphs, build_mol_composition_cache
from src.data.audio import build_audio_feature_cache
from src.data.text_embed import build_text_embedding_cache
from src.data.nmr import build_nmr_feature_cache
from src.data.image_cache import build_image_feature_cache
from src.data.augment import TrainingAugmentation
from src.models.utopya import UTOPYAModel, count_parameters
from src.training.loss import UTOPYALoss, LAMBDA_PRED, LAMBDA_CLS, LAMBDA_PHYS
from src.training.train import pretrain_tcn, train_utopya
from src.evaluation.evaluate import evaluate_utopya, evaluate_baselines, print_comparison
from src.evaluation.metrics import multisignal_auroc_paper, find_optimal_multisignal_weight


DATA_ROOT  = config.DATA_ROOT
SYSTEM     = config.SYSTEM


def _bool_flag(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes")


def parse_args():
    p = argparse.ArgumentParser(description="UTOPYA training + evaluation")
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--train-epochs",    type=int, default=80)
    p.add_argument("--batch-size",      type=int, default=64)
    p.add_argument("--device",          type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir",        type=str, default=None)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--skip-pretrain",   action="store_true", help="Skip TCN pretraining")
    p.add_argument("--skip-baselines",  action="store_true", help="Skip baseline evaluation")
    p.add_argument("--results-json",    type=str,
                   default=os.path.join(config.PACKAGE_ROOT, "results.json"))
    p.add_argument("--ablation",        type=str, default=None, help="Ablation config (A1-A11); if None, runs full multimodal (A7)")
    # --- Review workflow additions (additive; defaults reproduce prior behaviour) ---
    p.add_argument("--seed",            type=int, default=None,
                   help="Seed model init / training randomness (does not affect the "
                        "leak-free split search, which has its own fixed seed=42).")
    p.add_argument("--run-id",          type=str, default=None,
                   help="If set, also writes outputs/{run_id}/metrics.json and defaults "
                        "--save-dir to outputs/{run_id}/checkpoints.")
    p.add_argument("--smoothness-loss", type=_bool_flag, default=True,
                   help="Enable the temporal-smoothness physics term (true/false).")
    p.add_argument("--mono-loss",       type=_bool_flag, default=True,
                   help="Enable the temperature-monotonicity physics term (true/false).")
    p.add_argument("--physics-loss",    type=_bool_flag, default=True,
                   help="Master switch for the physics loss term (lambda_phys); if "
                        "false, overrides --smoothness-loss/--mono-loss to zero.")
    p.add_argument("--curriculum",      type=_bool_flag, default=True,
                   help="Enable curriculum (difficulty-weighted) sampling (true/false).")
    p.add_argument("--pretrained",      type=_bool_flag, default=True,
                   help="Enable TCN self-supervised pretraining (true/false); "
                        "false is equivalent to --skip-pretrain.")
    p.add_argument("--lambda-mono",     type=float, default=1.0,
                   help="Weight of the monotonicity term inside the physics loss.")
    p.add_argument("--wpred",           type=float, default=LAMBDA_PRED)
    p.add_argument("--wclass",          type=float, default=LAMBDA_CLS)
    p.add_argument("--wphys",           type=float, default=LAMBDA_PHYS)
    # --- Procedure-fidelity additions (PROCEDURES_AUDIT.md) ---
    p.add_argument("--focal-loss",       type=_bool_flag, default=True,
                   help="Use focal loss for classification (true) or plain "
                        "weighted cross-entropy (false) — reconstructs the "
                        "paper's 'Remove focal loss' ablation arm.")
    p.add_argument("--uncertainty-weighting", type=_bool_flag, default=False,
                   help="Learn Kendall-et-al. homoscedastic-uncertainty task "
                        "weights instead of the fixed wpred/wclass/wphys.")
    p.add_argument("--augment",         type=_bool_flag, default=True,
                   help="Apply training-time jitter/scaling/time-warp "
                        "augmentation to the training split (true/false) — "
                        "previously always a no-op regardless of setting.")
    p.add_argument("--freeze-epochs",   type=int, default=3,
                   help="Epochs the pretrained TCN encoder stays frozen "
                        "before fine-tuning (paper: 3).")
    p.add_argument("--tcn-lr-scale",    type=float, default=0.1,
                   help="LR multiplier for the TCN encoder once unfrozen, "
                        "relative to the base --lr (paper: 0.1, i.e. 10x smaller).")
    p.add_argument("--modality-dropout-p", type=float, default=0.2,
                   help="Per-modality independent Bernoulli dropout "
                        "probability applied during training only (paper: 0.2).")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve --run-id-dependent defaults (kept out of argparse so plain
    # `python src/run.py` behaves exactly as before).
    if args.save_dir is None:
        args.save_dir = (os.path.join(config.OUTPUTS_ROOT, args.run_id, "checkpoints")
                         if args.run_id else os.path.join(config.CHECKPOINTS_ROOT, "A7"))

    print(f"Device: {args.device}")
    print(f"Save dir: {args.save_dir}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Load ablation config if specified
    # ------------------------------------------------------------------
    from src.ablations import get_ablation_config
    ablation_config = get_ablation_config(args.ablation) if args.ablation else get_ablation_config("A7")
    ablation_name = args.ablation or "A7"
    print(f"\nAblation: {ablation_name} — {ablation_config.description}")
    if ablation_config.notes:
        print(f"  Note: {ablation_config.notes}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    t0 = time.time()
    all_exps = load_all_experiments(DATA_ROOT)
    metas    = [m for _, _, m in all_exps]
    print(f"  Loaded {len(all_exps)} experiments in {time.time()-t0:.1f}s")
    exp_meta_list = [{"op": m.operating_point, "name": m.experiment_name} for m in metas]

    # ------------------------------------------------------------------
    # 2. Static feature caches (tabular, GC molar composition, audio, text)
    # ------------------------------------------------------------------
    print("\n[2/6] Building static feature caches...")
    tab_cache = build_static_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    mol_comp_cache = build_mol_composition_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    audio_cache = build_audio_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    text_cache = build_text_embedding_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    # NMR/image caches are only built when this ablation config actually uses
    # them (Section 5.9 A15/A14) — building the image cache decodes video for
    # every experiment, which is unnecessary cost for configs that don't use it.
    nmr_cache = (build_nmr_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
                 if ablation_config.use_nmr else None)
    image_cache = (build_image_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list,
                                              device=args.device)
                   if ablation_config.use_image else None)

    # ------------------------------------------------------------------
    # 3. Splits + Datasets
    # ------------------------------------------------------------------
    print("\n[3/6] Building datasets...")
    train_idx, val_idx, test_idx = find_leak_free_split(metas)
    train_exps = [all_exps[i] for i in train_idx]
    val_exps   = [all_exps[i] for i in val_idx]
    test_exps  = [all_exps[i] for i in test_idx]

    train_augment_fn = TrainingAugmentation() if args.augment else None

    def _make_ds(exps, augment_fn=None):
        return BatchDistillationDataset(
            exps, normalise=True,
            augment_fn=augment_fn,
            tab_feature_cache=tab_cache,
            mol_composition_cache=mol_comp_cache,
            audio_feature_cache=audio_cache,
            text_embedding_cache=text_cache,
            nmr_feature_cache=nmr_cache,
            image_feature_cache=image_cache,
        )

    # Augmentation (jitter/scaling/time-warp) is applied to the TRAINING
    # split only — val/test must reflect the un-augmented distribution.
    train_ds = _make_ds(train_exps, augment_fn=train_augment_fn)
    val_ds   = _make_ds(val_exps)
    test_ds  = _make_ds(test_exps)

    print(f"  {train_ds.summary()}")
    print(f"  Val:  {len(val_ds)} windows")
    print(f"  Test: {len(test_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 4. Pre-training
    # ------------------------------------------------------------------
    tcn_weights_path = os.path.join(args.save_dir, "tcn_pretrained.pt")
    skip_pretrain = args.skip_pretrain or not args.pretrained

    if not skip_pretrain:
        print("\n[4/6] TCN self-supervised pre-training...")
        pretrain_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        pretrain_model = pretrain_tcn(
            pretrain_loader,
            n_vars    = N_INPUT_VARS,
            dmodel    = 128,
            n_epochs  = args.pretrain_epochs,
            lr        = args.lr,
            device    = args.device,
            save_path = tcn_weights_path,
        )
    else:
        print("\n[4/6] Skipping pre-training")
        pretrain_model = None

    # ------------------------------------------------------------------
    # 5. Full model training
    # ------------------------------------------------------------------
    print("\n[5/6] Training full UTOPYA model...")
    model = UTOPYAModel(
        d_tab       = D_TAB,
        n_vars      = N_INPUT_VARS,
        n_cont      = N_CONTINUOUS,
        horizon     = 60,
        window      = WINDOW_SIZE,
        dmodel      = 128,
        use_tabular = ablation_config.use_tabular,
        use_text    = ablation_config.use_text,
        use_gc      = ablation_config.use_gc,
        use_audio   = ablation_config.use_audio,
        use_nmr     = ablation_config.use_nmr,
        use_image   = ablation_config.use_image,
        dropout     = 0.5,
    ).to(args.device)

    # Register molecule graphs if GC modality is active
    if ablation_config.use_gc:
        mol_graphs = build_molecule_graphs(SYSTEM, device=torch.device(args.device))
        if mol_graphs is not None:
            model.register_molecule_graphs(mol_graphs)
            print(f"  Registered {len(mol_graphs)} molecule graphs in model.")
        else:
            print("  WARNING: Could not build molecule graphs — GC modality will use zeros.")

    # Load pretrained TCN weights if available
    if pretrain_model is not None:
        model.tcn.load_state_dict(pretrain_model.encoder.state_dict())
        print("  Loaded pretrained TCN weights.")
    elif os.path.exists(tcn_weights_path):
        model.tcn.load_state_dict(torch.load(tcn_weights_path, map_location=args.device))
        print(f"  Loaded TCN weights from {tcn_weights_path}")

    print(f"  Model parameters: {count_parameters(model):,}")

    # Physics loss: temperature monotone column indices within N_CONTINUOUS
    from src.data.dataset import CONTINUOUS_COL_INDICES
    from src.data.loader import ALL_COLS
    MONO_COLS = ["T703", "T709", "T711", "T712"]   # expected non-increasing
    mono_local = [
        CONTINUOUS_COL_INDICES.index(ALL_COLS.index(c))
        for c in MONO_COLS
        if c in ALL_COLS and ALL_COLS.index(c) in CONTINUOUS_COL_INDICES
    ]

    effective_wphys = args.wphys if args.physics_loss else 0.0
    loss_fn = UTOPYALoss(
        mono_col_indices=mono_local,
        lambda_pred=args.wpred,
        lambda_cls=args.wclass,
        lambda_phys=effective_wphys,
        lambda_smooth_phys=1.0 if args.smoothness_loss else 0.0,
        lambda_mono_phys=args.lambda_mono if args.mono_loss else 0.0,
        use_focal=args.focal_loss,
        learnable_weights=args.uncertainty_weighting,
    )
    print(f"  Loss weights: wpred={args.wpred} wclass={args.wclass} "
          f"wphys={effective_wphys} (smoothness={args.smoothness_loss} "
          f"mono={args.mono_loss} lambda_mono={args.lambda_mono}) "
          f"curriculum={args.curriculum} pretrained={args.pretrained} "
          f"focal_loss={args.focal_loss} uncertainty_weighting={args.uncertainty_weighting} "
          f"augment={args.augment} freeze_epochs={args.freeze_epochs} "
          f"tcn_lr_scale={args.tcn_lr_scale} modality_dropout_p={args.modality_dropout_p}")

    history = train_utopya(
        model          = model,
        train_dataset  = train_ds,
        val_dataset    = val_ds,
        loss_fn        = loss_fn,
        n_epochs       = args.train_epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = 1e-3,
        device         = args.device,
        save_dir       = args.save_dir,
        freeze_tcn_epochs   = args.freeze_epochs,
        tcn_lr_scale        = args.tcn_lr_scale,
        modality_dropout_p  = args.modality_dropout_p,
        curriculum_warmup   = 20,
        curriculum          = args.curriculum,
    )

    # Load best checkpoint for evaluation
    best_ckpt = os.path.join(args.save_dir, "utopya_best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n  Loaded best checkpoint (epoch {ckpt['epoch']}, val={ckpt['val_loss']:.4f})")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluation...")
    utopya_metrics, test_results = evaluate_utopya(
        model, test_loader, device=args.device, split="test", return_raw=True
    )

    # Paper-faithful multi-signal fusion (Section 5.8): weight optimised on
    # the VALIDATION split, then applied to test — see PROCEDURES_AUDIT.md.
    _, val_results = evaluate_utopya(
        model, val_loader, device=args.device, split="val", return_raw=True
    )
    best_w, val_ms_auroc = find_optimal_multisignal_weight(val_results)
    test_ms_auroc_paper = multisignal_auroc_paper(test_results, weight=best_w)
    utopya_metrics["multisignal_auroc_paper"] = test_ms_auroc_paper
    utopya_metrics["multisignal_weight_optimal"] = best_w
    print(f"  Paper-faithful multi-signal: weight={best_w:.2f} "
          f"(val AUROC={val_ms_auroc:.4f}) -> test AUROC={test_ms_auroc_paper:.4f}")

    if not args.skip_baselines:
        print("\n  Running baselines...")
        baseline_metrics = evaluate_baselines(train_loader, test_loader, device=args.device)
        print_comparison(utopya_metrics, baseline_metrics)
    else:
        baseline_metrics = {}
        print("\n  (baselines skipped)")

    # ------------------------------------------------------------------
    # Review-workflow metrics dump (additive; only when --run-id is set)
    # ------------------------------------------------------------------
    if args.run_id:
        metrics_out = {
            "window_auroc": utopya_metrics.get("window_auroc"),
            "experiment_auroc": utopya_metrics.get("experiment_auroc"),
            "multisignal_auroc": utopya_metrics.get("multisignal_auroc"),
            "multisignal_auroc_paper": utopya_metrics.get("multisignal_auroc_paper"),
            "multisignal_weight_optimal": utopya_metrics.get("multisignal_weight_optimal"),
            "auprc": utopya_metrics.get("window_auprc"),
            "f1": utopya_metrics.get("f1_optimal"),
            "mae": utopya_metrics.get("pred_mae_mean"),
        }
        metrics_dir = os.path.join(config.OUTPUTS_ROOT, args.run_id)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2, default=str)
        print(f"  Review-workflow metrics written to {metrics_dir}/metrics.json")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "utopya": {k: v for k, v in utopya_metrics.items() if k != "pred_mae_per_var"},
        "baselines": {
            name: {k: v for k, v in m.items() if k != "pred_mae_per_var"}
            for name, m in baseline_metrics.items()
        },
        "history": history,
    }
    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.results_json}")


if __name__ == "__main__":
    main()
