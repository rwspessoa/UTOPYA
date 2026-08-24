"""
Molecular graph builder and GC composition cache for UTOPYA (Bug 2 fix).

The paper's GC modality uses a K=3-layer GCN to encode the molecular identity
of each chemical component, then weights the per-molecule embeddings by the
experiment's mean molar fractions from GC chromatography data:

    z_mol = Σ x_i · GCN(molecule_i)

The GCN is trained END-TO-END.  To enable proper gradient flow the molecule
graphs are stored as registered buffers in the model and the weighted
combination is computed inside the model's forward pass.

This module provides two utilities:
  1. build_molecule_graphs(system, device)
        Returns a list of (node_feats, adj_norm) tensors for the fixed
        molecules in the system.  Call model.register_molecule_graphs() with
        this list before training.

  2. build_mol_composition_cache(data_root, system, experiments)
        Returns a dict (op, name) → np.ndarray (n_mol,) of mean molar
        fractions read from the GC tabular CSV files.  Passed to
        BatchDistillationDataset via mol_composition_cache.

Usage:
    from src.data.molecular import build_molecule_graphs, build_mol_composition_cache

    graphs = build_molecule_graphs(SYSTEM, device=torch.device(args.device))
    if graphs is not None:
        model.register_molecule_graphs(graphs)

    mol_cache = build_mol_composition_cache(DATA_ROOT, SYSTEM, exp_meta_list)
    train_ds = BatchDistillationDataset(..., mol_composition_cache=mol_cache)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.encoders import build_molecule_graph


# ---------------------------------------------------------------------------
# Chemical system registry
# ---------------------------------------------------------------------------

# SMILES for each component of the system (order MUST match GC CSV column order).
SYSTEM_SMILES: Dict[str, List[str]] = {
    "batch_dist_ternary_butan-1-ol+propan-2-ol+water": [
        "CC(C)O",   # propan-2-ol (isopropanol)
        "CCCCO",    # butan-1-ol  (1-butanol)
        "O",        # water
    ],
}

# GC CSV column names for mean molar fractions (must be in same order as SYSTEM_SMILES).
GC_MOL_FRACTION_COLS: Dict[str, List[str]] = {
    "batch_dist_ternary_butan-1-ol+propan-2-ol+water": [
        "propan-2-ol_mol mean",
        "butan-1-ol_mol mean",
        "water_mol mean",
    ],
}

GC_ROOT_DIRNAME = "10_Batch_Distillation_Plant_M-202210_GC_Composition"


# ---------------------------------------------------------------------------
# 1. Molecule graphs (fixed per chemical system)
# ---------------------------------------------------------------------------

def build_molecule_graphs(
    system: str,
    device: torch.device,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Build (node_feats, adj_norm) tuples for every molecule in `system`.

    Returns a list of length n_mol, or None if the system is unknown or
    RDKit is not installed.  Each tuple contains:
      - node_feats : (N_atoms, N_ATOM_FEATURES) float32
      - adj_norm   : (N_atoms, N_atoms) float32  symmetrically normalised adj
    """
    smiles_list = SYSTEM_SMILES.get(system)
    if smiles_list is None:
        print(f"[Molecular] System '{system}' not in registry — skipping GC modality.")
        return None

    graphs = []
    for smiles in smiles_list:
        graph = build_molecule_graph(smiles, device=device)
        if graph is None:
            print(f"[Molecular] RDKit unavailable — cannot build graph for '{smiles}'.")
            return None
        graphs.append(graph)

    print(f"[Molecular] Built {len(graphs)} molecule graphs for system '{system}'.")
    return graphs


# ---------------------------------------------------------------------------
# 2. Per-experiment GC molar fractions
# ---------------------------------------------------------------------------

def _load_molar_fractions(
    data_root: str,
    system:    str,
    op:        str,
    name:      str,
    mol_cols:  List[str],
) -> Optional[np.ndarray]:
    """
    Read and normalise mean molar fractions (n_mol,) from the GC CSV.
    Returns None if the CSV does not exist.
    """
    gc_dir   = os.path.join(data_root, GC_ROOT_DIRNAME, "tabular", system, op)
    csv_path = os.path.join(gc_dir, f"{name}.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    n  = len(mol_cols)
    fracs = []
    for col in mol_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            fracs.append(float(vals.mean()) if len(vals) > 0 else 1.0 / n)
        else:
            fracs.append(1.0 / n)

    arr   = np.array(fracs, dtype=np.float32)
    total = arr.sum()
    return arr / total if total > 1e-6 else np.ones(n, dtype=np.float32) / n


def build_mol_composition_cache(
    data_root:   str,
    system:      str,
    experiments: List[Dict],   # dicts with keys "op" and "name"
) -> Optional[Dict[Tuple[str, str], np.ndarray]]:
    """
    Build per-experiment molar fraction vectors from GC tabular data.

    Returns
    -------
    dict : (op, name) → np.ndarray (n_mol,) normalised molar fractions
        None if the system is not in the registry.

    Experiments for which no GC CSV exists fall back to equal weights.
    """
    mol_cols = GC_MOL_FRACTION_COLS.get(system)
    if mol_cols is None:
        print(f"[Molecular] System '{system}' not in GC registry — skipping composition cache.")
        return None

    n        = len(mol_cols)
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing  = 0

    for meta in experiments:
        op   = meta["op"]
        name = meta["name"]
        fracs = _load_molar_fractions(data_root, system, op, name, mol_cols)
        if fracs is None:
            fracs = np.ones(n, dtype=np.float32) / n   # equal-weight fallback
            missing += 1
        cache[(op, name)] = fracs

    if missing:
        print(
            f"[Molecular] {missing}/{len(experiments)} experiments used equal-weight "
            f"fallback (GC CSV not found)."
        )
    print(f"[Molecular] Built composition cache for {len(cache)} experiments ({n} components).")
    return cache
