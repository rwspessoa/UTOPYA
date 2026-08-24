"""
Shared setup for scripts that need the trained A7 (headline, full-multimodal)
model plus the full dataset/split/caches — a self-contained equivalent of the
review scaffold's utils/utopya_bridge.py, with no scaffold dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

import config
from src.data.loader import load_all_experiments
from src.data.splits import find_leak_free_split
from src.data.tabular import build_static_feature_cache, D_TAB
from src.data.molecular import build_molecule_graphs, build_mol_composition_cache
from src.data.audio import build_audio_feature_cache
from src.data.text_embed import build_text_embedding_cache
from src.data.dataset import N_INPUT_VARS, N_CONTINUOUS, WINDOW_SIZE
from src.models.utopya import UTOPYAModel


def load_model_and_data(load_checkpoint: bool = True) -> dict:
    """
    Returns a dict with: model, device, all_exps, tab_cache, mol_cache,
    audio_cache, text_cache, train_idx, val_idx, test_idx, ckpt_path.

    If load_checkpoint is False, the model is returned with random
    initialisation (e.g. when the caller will load a different checkpoint
    itself, as train_frozen_extension.py and train_reconstruction_head.py do).
    """
    config.ensure_dirs()
    device = config.DEVICE

    all_exps = load_all_experiments(config.DATA_ROOT, verbose=False)
    metas = [m for _, _, m in all_exps]
    exp_meta_list = [{"op": m.operating_point, "name": m.experiment_name} for m in metas]

    tab_cache = build_static_feature_cache(config.DATA_ROOT, config.SYSTEM, exp_meta_list)
    mol_cache = build_mol_composition_cache(config.DATA_ROOT, config.SYSTEM, exp_meta_list)
    audio_cache = build_audio_feature_cache(config.DATA_ROOT, config.SYSTEM, exp_meta_list)
    text_cache = build_text_embedding_cache(config.DATA_ROOT, config.SYSTEM, exp_meta_list)

    train_idx, val_idx, test_idx = find_leak_free_split(metas)

    model = UTOPYAModel(
        d_tab=D_TAB, n_vars=N_INPUT_VARS, n_cont=N_CONTINUOUS,
        horizon=60, window=WINDOW_SIZE, dmodel=128,
        use_tabular=True, use_text=True, use_gc=True, use_audio=True,
        dropout=0.0,
    ).to(device)

    graphs = build_molecule_graphs(config.SYSTEM, device=torch.device(device))
    if graphs is not None:
        model.register_molecule_graphs(graphs)

    ckpt_path = None
    if load_checkpoint:
        ckpt_path = config.find_a7_checkpoint()
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"Loaded checkpoint {ckpt_path}")

    return {
        "model": model, "device": device, "all_exps": all_exps, "metas": metas,
        "tab_cache": tab_cache, "mol_cache": mol_cache, "audio_cache": audio_cache,
        "text_cache": text_cache,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "ckpt_path": ckpt_path,
    }
