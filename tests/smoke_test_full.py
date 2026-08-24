"""
End-to-end smoke test for the full UTOPYA pipeline, run against the Zenodo
dataset (Arweiler et al. 2026).

Tests (in order):
  1. Data loading + tabular feature extraction
  2. Dataset + DataLoader with tabular features
  3. UTOPYAModel forward pass
  4. UTOPYALoss backward pass
  5. Gradient flow check

Run from project root:
    python -m tests.smoke_test_full
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config

DEVICE = config.DEVICE
print(f"Device: {DEVICE}")

# -------------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------------
print("\n[1/5] Loading experiments...")
t0 = time.time()
from src.data.loader import load_all_experiments
all_exps = load_all_experiments(config.DATA_ROOT)
print(f"  Loaded {len(all_exps)} experiments in {time.time()-t0:.1f}s")

# -------------------------------------------------------------------------
# 2. Tabular feature cache
# -------------------------------------------------------------------------
print("\n[2/5] Building tabular feature cache...")
from src.data.tabular import build_static_feature_cache, D_TAB
metas = [m for _, _, m in all_exps]
tab_cache = build_static_feature_cache(
    data_root  = config.DATA_ROOT,
    system     = config.SYSTEM,
    experiments= [{"op": m.operating_point, "name": m.experiment_name} for m in metas],
)
# Quick sanity check
sample_key = list(tab_cache.keys())[0]
sample_feat = tab_cache[sample_key]
print(f"  d_tab={D_TAB}, sample key={sample_key}, feat shape={sample_feat.shape}")
print(f"  feat values: {sample_feat}")

# -------------------------------------------------------------------------
# 3. Dataset + DataLoader
# -------------------------------------------------------------------------
print("\n[3/5] Building dataset...")
from src.data.splits import find_leak_free_split
from src.data.dataset import BatchDistillationDataset
from torch.utils.data import DataLoader

train_idx, val_idx, _ = find_leak_free_split(metas)
train_exps = [all_exps[i] for i in train_idx]
val_exps   = [all_exps[i] for i in val_idx[:5]]  # small val for speed

train_ds = BatchDistillationDataset(train_exps, normalise=True, tab_feature_cache=tab_cache)
val_ds   = BatchDistillationDataset(val_exps,   normalise=True, tab_feature_cache=tab_cache)
print(f"  {train_ds.summary()}")

loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"  batch keys: {list(batch.keys())}")
print(f"  x: {batch['x'].shape}  y_target: {batch['y_target'].shape}  "
      f"tab: {batch['tab'].shape}  label_seq: {batch['label_seq'].shape}")

# -------------------------------------------------------------------------
# 4. Full UTOPYA model forward pass
# -------------------------------------------------------------------------
print("\n[4/5] Building UTOPYA model and forward pass...")
from src.models.utopya import UTOPYAModel, count_parameters
from src.data.dataset import N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE

model = UTOPYAModel(
    d_tab   = D_TAB,
    n_vars  = N_INPUT_VARS,
    n_cont  = N_CONTINUOUS,
    horizon = 60,
    window  = WINDOW_SIZE,
    dmodel  = 128,
    use_text= False,   # skip SBERT for smoke test
    use_mol = False,
    dropout = 0.5,
).to(DEVICE)
print(f"  Parameters: {count_parameters(model):,}")

x_ts  = batch["x"].to(DEVICE)
x_tab = batch["tab"].to(DEVICE)
with torch.no_grad():
    y_hat, cls_logits, loc_logits = model(x_ts, x_tab)
print(f"  y_hat:       {y_hat.shape}         (B, H, N_cont)")
print(f"  cls_logits:  {cls_logits.shape}    (B, 2)")
print(f"  loc_logits:  {loc_logits.shape}    (B, T, N_vars)")

# -------------------------------------------------------------------------
# 5. Loss backward pass
# -------------------------------------------------------------------------
print("\n[5/5] Loss + backward pass...")
from src.training.loss import UTOPYALoss

loss_fn = UTOPYALoss()

labels    = batch["label_seq"].to(DEVICE)   # (B, W)
cls_label = (labels.max(dim=1).values > 0).long()
loc_label = (labels > 0).float().unsqueeze(-1).expand(-1, -1, N_INPUT_VARS)
y_true    = batch["y_target"].to(DEVICE)

# Need grad for backward
model.train()
y_hat, cls_logits, loc_logits = model(x_ts, x_tab)
losses = loss_fn(y_hat, y_true, cls_logits, cls_label, loc_logits, loc_label)
losses["total"].backward()
print(f"  total={losses['total'].item():.4f}  "
      f"pred={losses['pred'].item():.4f}  "
      f"cls={losses['cls'].item():.4f}  "
      f"loc={losses['loc'].item():.4f}  "
      f"phys={losses['phys'].item():.4f}")

# Check gradients flow
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print(f"  Params with gradients: {len(grad_norms)} / {len(list(model.parameters()))}")
print(f"  Max grad norm: {max(grad_norms):.4f}")

print("\nAll UTOPYA smoke tests passed!")
