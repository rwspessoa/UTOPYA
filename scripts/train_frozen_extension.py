"""
Frozen-backbone modality extension (paper Section 5.9 / Appendix A14-A15).

Reconstructs the paper's "Marginal Contribution of NMR and Image Modalities"
experiment: load the trained A7 checkpoint, freeze every parameter that
already existed in A7 (TCN, audio/tab/text/GC encoders, FiLM, cross-modal
attention, gated-fusion gate), graft ONE new dynamic-modality encoder onto
the architecture (NMR -> "A15", Image -> "A14"), and fine-tune only the new
encoder + the gated-fusion gate + the task heads for 10 epochs at a small
learning rate (5e-5), no curriculum — exactly the protocol the paper
describes. This experiment previously had zero corresponding implementation
in this repo (see PROCEDURES_AUDIT.md).

Usage:
    python scripts/train_frozen_extension.py --modality nmr
    python scripts/train_frozen_extension.py --modality image
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from src.data.loader import load_all_experiments, ALL_COLS
from src.data.splits import find_leak_free_split
from src.data.dataset import BatchDistillationDataset, N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE
from src.data.tabular import build_static_feature_cache, D_TAB
from src.data.molecular import build_molecule_graphs, build_mol_composition_cache
from src.data.audio import build_audio_feature_cache
from src.data.text_embed import build_text_embedding_cache
from src.data.nmr import build_nmr_feature_cache
from src.data.image_cache import build_image_feature_cache
from src.models.utopya import UTOPYAModel, count_parameters
from src.evaluation.evaluate import evaluate_utopya

DATA_ROOT = config.DATA_ROOT
SYSTEM = config.SYSTEM
find_a7_checkpoint = config.find_a7_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["nmr", "image"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config_name = "A14" if args.modality == "image" else "A15"
    print(f"=== Frozen-backbone extension: {config_name} ({args.modality}) ===")

    print("[1/5] Loading data + caches...")
    all_exps = load_all_experiments(DATA_ROOT, verbose=False)
    metas = [m for _, _, m in all_exps]
    exp_meta_list = [{"op": m.operating_point, "name": m.experiment_name} for m in metas]

    tab_cache = build_static_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    mol_cache = build_mol_composition_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    audio_cache = build_audio_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    text_cache = build_text_embedding_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    nmr_cache = build_nmr_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list) if args.modality == "nmr" else None
    image_cache = build_image_feature_cache(DATA_ROOT, SYSTEM, exp_meta_list, device=args.device) if args.modality == "image" else None

    train_idx, val_idx, test_idx = find_leak_free_split(metas)
    train_exps = [all_exps[i] for i in train_idx]
    val_exps = [all_exps[i] for i in val_idx]
    test_exps = [all_exps[i] for i in test_idx]

    def make_ds(exps):
        return BatchDistillationDataset(
            exps, normalise=True,
            tab_feature_cache=tab_cache, mol_composition_cache=mol_cache,
            audio_feature_cache=audio_cache, text_embedding_cache=text_cache,
            nmr_feature_cache=nmr_cache, image_feature_cache=image_cache,
        )

    train_ds, val_ds, test_ds = make_ds(train_exps), make_ds(val_exps), make_ds(test_exps)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)} windows")

    print("[2/5] Building extended model (A7 + new modality encoder)...")
    model = UTOPYAModel(
        d_tab=D_TAB, n_vars=N_INPUT_VARS, n_cont=N_CONTINUOUS,
        horizon=60, window=WINDOW_SIZE, dmodel=128,
        use_tabular=True, use_text=True, use_gc=True, use_audio=True,
        use_nmr=(args.modality == "nmr"), use_image=(args.modality == "image"),
        dropout=0.5,
    ).to(args.device)

    if model.use_gc:
        graphs = build_molecule_graphs(SYSTEM, device=torch.device(args.device))
        if graphs is not None:
            model.register_molecule_graphs(graphs)

    ckpt_path = find_a7_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=args.device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded A7 backbone from {ckpt_path}")
    print(f"  New (randomly-initialised) params: {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored, from a differently-shaped checkpoint): {unexpected}")

    # --- Freeze every pre-existing A7 parameter; train only the new branch ---
    trainable_modules = ["gate_proj", "pred_head", "cls_head", "loc_head", "phase_head"]
    trainable_modules.append("nmr_enc" if args.modality == "nmr" else "image_enc")
    n_total = sum(p.numel() for p in model.parameters())
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(m + ".") or name == m for m in trainable_modules)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} total "
          f"(modules: {trainable_modules})")

    print(f"[3/5] Fine-tuning for {args.epochs} epochs, lr={args.lr}, no curriculum...")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-3,
    )
    best_val_auroc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen submodules in eval mode (no BN/dropout stat updates there);
        # only the trainable ones need train() behaviour.
        for name, m in model.named_modules():
            m.eval()
        for tm in trainable_modules:
            getattr(model, tm).train()

        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            x_ts = batch["x"].to(args.device)
            y_true = batch["y_target"].to(args.device)
            labels = batch["label_seq"].to(args.device)
            x_tab = batch["tab"].to(args.device)
            x_text = batch.get("x_text")
            x_text = x_text.to(args.device) if x_text is not None else None
            mol_comp = batch.get("mol_composition")
            mol_comp = mol_comp.to(args.device) if mol_comp is not None else None
            x_audio = batch.get("x_audio")
            x_audio = x_audio.to(args.device) if x_audio is not None else None
            x_nmr = batch.get("x_nmr")
            x_nmr = x_nmr.to(args.device) if x_nmr is not None else None
            x_image = batch.get("x_image")
            x_image = x_image.to(args.device) if x_image is not None else None

            cls_label = (labels.max(dim=1).values > 0).long()

            opt.zero_grad()
            y_hat, cls_logits, _ = model(
                x_ts, x_tab, x_text=x_text, mol_composition=mol_comp,
                x_audio=x_audio, x_nmr=x_nmr, x_image=x_image,
            )
            loss = (
                0.1 * F.mse_loss(y_hat, y_true) +
                2.0 * F.cross_entropy(cls_logits, cls_label)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate_utopya(model, val_loader, device=args.device, split="val")
        elapsed = time.time() - t0
        print(f"  [epoch {epoch:2d}/{args.epochs}] train_loss={total_loss/max(1,n_batches):.4f} "
              f"val_auroc={val_metrics['window_auroc']:.4f}  ({elapsed:.1f}s)")
        if val_metrics["window_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["window_auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"[4/5] Best val AUROC: {best_val_auroc:.4f}")

    print("[5/5] Evaluating on test split...")
    test_metrics = evaluate_utopya(model, test_loader, device=args.device, split="test")

    out_dir = os.path.join(config.CHECKPOINTS_ROOT, config_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "val_auroc": best_val_auroc},
               os.path.join(out_dir, "utopya_best.pt"))
    metrics_out = {k: v for k, v in test_metrics.items() if k != "pred_mae_per_var"}
    metrics_out["val_auroc"] = best_val_auroc
    metrics_out["config"] = config_name
    metrics_out["modality"] = args.modality
    metrics_out["a7_checkpoint_used"] = ckpt_path
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"Saved checkpoint + metrics to {out_dir}")


if __name__ == "__main__":
    main()
