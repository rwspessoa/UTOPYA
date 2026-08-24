"""
Train the post-hoc PhaseClassificationHead on top of the frozen A7 backbone.

Not part of the original architecture — added to reconstruct the paper's
Figure 5 (4-class phase confusion matrix), which has no corresponding head
in the current codebase (see conversation / ADAPTATIONS.md). The backbone
(TCN, static encoders, FiLM, cross-attention, gated fusion) is frozen;
only phase_head's ~16.9k params are trained, using BatchDistillationDataset's
existing per-window `phase_label` (0=normal,1=blind,2=anomalous,3=recovery)
as the target — no new labels invented, this field already exists.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from common import load_model_and_data
from src.data.dataset import BatchDistillationDataset

ctx = load_model_and_data()
model, device = ctx["model"], ctx["device"]

train_exps = [ctx["all_exps"][i] for i in ctx["train_idx"]]
val_exps = [ctx["all_exps"][i] for i in ctx["val_idx"]]


def make_ds(exps):
    return BatchDistillationDataset(
        exps, normalise=True,
        tab_feature_cache=ctx["tab_cache"],
        mol_composition_cache=ctx["mol_cache"],
        audio_feature_cache=ctx["audio_cache"],
        text_embedding_cache=ctx["text_cache"],
    )


train_ds = make_ds(train_exps)
val_ds = make_ds(val_exps)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")

# Freeze everything except phase_head
for p in model.parameters():
    p.requires_grad = False
for p in model.phase_head.parameters():
    p.requires_grad = True

opt = torch.optim.AdamW(model.phase_head.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

N_EPOCHS = 15
best_val_acc = 0.0
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    model.eval()          # backbone stays in eval mode (no BN/dropout updates)
    model.phase_head.train()
    total_loss, n_batches = 0.0, 0
    for batch in train_loader:
        x_ts = batch["x"].to(device)
        x_tab = batch["tab"].to(device)
        x_text = batch.get("x_text")
        x_text = x_text.to(device) if x_text is not None else None
        x_audio = batch.get("x_audio")
        x_audio = x_audio.to(device) if x_audio is not None else None
        mol_comp = batch.get("mol_composition")
        mol_comp = mol_comp.to(device) if mol_comp is not None else None
        phase_label = batch["phase_label"].to(device)

        opt.zero_grad()
        with torch.no_grad():
            # backbone frozen: no grad needed except phase_head itself
            pass
        _, _, _, phase_logits = model(
            x_ts, x_tab, x_text=x_text, mol_composition=mol_comp, x_audio=x_audio,
            return_phase=True,
        )
        loss = loss_fn(phase_logits, phase_label)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x_ts = batch["x"].to(device)
            x_tab = batch["tab"].to(device)
            x_text = batch.get("x_text")
            x_text = x_text.to(device) if x_text is not None else None
            x_audio = batch.get("x_audio")
            x_audio = x_audio.to(device) if x_audio is not None else None
            mol_comp = batch.get("mol_composition")
            mol_comp = mol_comp.to(device) if mol_comp is not None else None
            phase_label = batch["phase_label"].to(device)

            _, _, _, phase_logits = model(
                x_ts, x_tab, x_text=x_text, mol_composition=mol_comp, x_audio=x_audio,
                return_phase=True,
            )
            pred = phase_logits.argmax(dim=-1)
            correct += (pred == phase_label).sum().item()
            total += len(phase_label)
    val_acc = correct / max(1, total)
    print(f"[epoch {epoch:2d}/{N_EPOCHS}] train_loss={total_loss/max(1,n_batches):.4f} "
          f"val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.phase_head.state_dict().items()}

model.phase_head.load_state_dict(best_state)
print(f"Best val_acc: {best_val_acc:.4f}")

# Save full model (backbone + trained phase_head) as a new checkpoint
out_dir = os.path.join(config.CHECKPOINTS_ROOT, "A7_v2")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "utopya_best_with_phase_head.pt")
torch.save({"model_state_dict": model.state_dict(), "phase_val_acc": best_val_acc}, out_path)
print(f"Saved to {out_path}")
