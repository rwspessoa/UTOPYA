"""
Smoke test: data pipeline + TCN encoder + pretrainer.

Verifies that the Zenodo dataset (Arweiler et al. 2026) can be loaded and
fed through the UTOPYA data pipeline and TCN encoder/pretrainer.

Run from project root:
    python -m tests.smoke_test
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    print("=" * 60)
    print("UTOPYA smoke test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load experiments
    # ------------------------------------------------------------------
    print("\n[1] Loading experiments...")
    from src.data.loader import load_all_experiments
    t0 = time.time()
    all_experiments = load_all_experiments(config.DATA_ROOT, verbose=True)
    print(f"    Done in {time.time() - t0:.1f}s — {len(all_experiments)} experiments")

    # Check data shapes
    data, labels, meta = all_experiments[0]
    print(f"    First experiment: {meta.operating_point}/{meta.experiment_name}")
    print(f"    data shape: {data.shape}, labels shape: {labels.shape}")
    print(f"    is_anomalous: {meta.is_anomalous}, label_counts: {meta.label_counts}")

    # ------------------------------------------------------------------
    # 2. Compute split
    # ------------------------------------------------------------------
    print("\n[2] Computing leak-free split...")
    from src.data.splits import find_leak_free_split
    all_metas = [m for _, _, m in all_experiments]
    train_idx, val_idx, test_idx = find_leak_free_split(all_metas, n_seeds=500)

    # ------------------------------------------------------------------
    # 3. Build datasets
    # ------------------------------------------------------------------
    print("\n[3] Building datasets...")
    from src.data.dataset import BatchDistillationDataset, N_CONTINUOUS, N_INPUT_VARS

    train_exps = [all_experiments[i] for i in train_idx]
    val_exps   = [all_experiments[i] for i in val_idx]
    test_exps  = [all_experiments[i] for i in test_idx]

    # Curriculum: start with easiest 60% (difficulty <= 0.3)
    train_ds_easy = BatchDistillationDataset(train_exps, normalise=True, difficulty_threshold=0.3)
    train_ds_full = BatchDistillationDataset(train_exps, normalise=True)
    val_ds   = BatchDistillationDataset(val_exps,   normalise=True)
    test_ds  = BatchDistillationDataset(test_exps,  normalise=True)

    print(f"    Train (easy): {train_ds_easy.summary()}")
    print(f"    Train (full): {train_ds_full.summary()}")
    print(f"    Val:          {val_ds.summary()}")
    print(f"    Test:         {test_ds.summary()}")

    # ------------------------------------------------------------------
    # 4. DataLoader sanity check
    # ------------------------------------------------------------------
    print("\n[4] DataLoader sanity check...")
    loader = DataLoader(train_ds_full, batch_size=16, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"    x shape:          {batch['x'].shape}")        # (16, 120, 31)
    print(f"    y_target shape:   {batch['y_target'].shape}") # (16, 60, 25)
    print(f"    label:            {batch['label'].shape}")
    print(f"    phase_label:      {batch['phase_label'].shape}")
    print(f"    difficulty range: [{batch['difficulty'].min():.2f}, {batch['difficulty'].max():.2f}]")

    assert batch["x"].shape == (16, 120, N_INPUT_VARS), "Unexpected x shape"
    assert batch["y_target"].shape == (16, 60, N_CONTINUOUS), "Unexpected y_target shape"

    # ------------------------------------------------------------------
    # 5. TCN encoder forward pass
    # ------------------------------------------------------------------
    print("\n[5] TCN encoder forward pass...")
    from src.models.tcn import TCNEncoder

    device = torch.device(config.DEVICE)
    print(f"    Device: {device}")

    encoder = TCNEncoder(n_input_vars=N_INPUT_VARS, dmodel=128, n_layers=6).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"    TCN parameters: {n_params:,}")
    print(f"    Receptive field: {encoder.receptive_field()} timesteps")

    x = batch["x"].to(device)                          # (16, 120, 31)
    z_ts, Z_ts = encoder(x)
    print(f"    z_ts shape (pooled):  {z_ts.shape}")   # (16, 128)
    print(f"    Z_ts shape (per-ts):  {Z_ts.shape}")   # (16, 120, 128)

    assert z_ts.shape == (16, 128)
    assert Z_ts.shape == (16, 120, 128)

    # ------------------------------------------------------------------
    # 6. Pretrainer forward pass
    # ------------------------------------------------------------------
    print("\n[6] Self-supervised pretrainer forward pass...")
    from src.models.pretrain import TCNPretrainer

    pretrainer = TCNPretrainer(encoder=encoder).to(device)
    n_pretrain = sum(p.numel() for p in pretrainer.parameters())
    print(f"    Pretrainer parameters: {n_pretrain:,}")

    loss_total, loss_recon, loss_contrast = pretrainer(x)
    print(f"    loss_total:    {loss_total.item():.4f}")
    print(f"    loss_recon:    {loss_recon.item():.4f}")
    print(f"    loss_contrast: {loss_contrast.item():.4f}")

    assert loss_total.requires_grad

    # Backprop
    loss_total.backward()
    print("    Backward pass: OK")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
