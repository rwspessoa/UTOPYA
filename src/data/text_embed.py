"""
Text (operator log) embedding cache builder for UTOPYA (Section 3.2 "Free-text fields").

Wires the "text" dataset modality (07_..._Tabular_Operation_Logs, free-text
operator log entries with columns Time,Property,Value) into the 384-d
Sentence-BERT embeddings TextEncoder expects.

Reviewer 2 leakage question: "Were explicit fault descriptions, anomaly
timings and post-experiment observations excluded from the text inputs?"
Answer implemented here: yes — any log row whose Property/Value text
contains a leakage keyword (fault/anomaly/alarm/failure/maintenance/
shutdown/blind/recovery/abnormal/"post experiment"/"anomaly timing", case-
insensitive substring match — see LEAKAGE_KEYWORDS, kept in sync with
config/review_config.yaml leakage.text_keywords) is dropped before
embedding. Because text is applied as one static feature per experiment
(uniformly to every window, like tabular/GC features), any residual
outcome-revealing phrase not caught by the keyword list would still leak
into every window of that experiment — this filter is a defensible but
not exhaustive safeguard; Phase 3 of the review workflow quantifies its
coverage against a hand-labelled sample for author review.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OPLOG_ROOT_DIRNAME = "07_Batch_Distillation_Plant_M-202210_Tabular_Operation_Logs"

SBERT_DIM = 384
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

LEAKAGE_KEYWORDS = [
    "fault", "anomaly", "alarm", "failure", "maintenance", "shutdown",
    "post experiment", "post-experiment", "anomaly timing", "blind",
    "recovery", "abnormal",
]


def _row_is_leaky(property_: str, value: str) -> bool:
    text = f"{property_} {value}".lower()
    return any(kw in text for kw in LEAKAGE_KEYWORDS)


def load_experiment_text(
    data_root: str, system: str, op: str, name: str,
) -> Tuple[Optional[str], int, int]:
    """
    Read one experiment's operation-log CSV, drop leakage-keyword rows, and
    return (filtered_text, n_rows_total, n_rows_dropped). filtered_text is
    None if the CSV doesn't exist or nothing survives filtering.
    """
    path = os.path.join(data_root, OPLOG_ROOT_DIRNAME, system, op, f"{name}.csv")
    if not os.path.exists(path):
        return None, 0, 0

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"Property", "Value"}.issubset(df.columns):
        return None, 0, 0

    n_total = len(df)
    keep_mask = ~df.apply(
        lambda r: _row_is_leaky(r.get("Property", ""), r.get("Value", "")), axis=1
    )
    kept = df[keep_mask]
    n_dropped = n_total - len(kept)

    parts = [
        f"{p.strip()}: {v.strip()}" if v.strip() else p.strip()
        for p, v in zip(kept["Property"], kept["Value"])
        if p.strip()
    ]
    text = ". ".join(parts).strip()
    return (text if text else None), n_total, n_dropped


def build_text_embedding_cache(
    data_root: str,
    system: str,
    experiments: List[Dict],   # dicts with keys "op" and "name"
    model_name: str = SBERT_MODEL_NAME,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build per-experiment SBERT embeddings from leakage-filtered operator logs.

    Returns dict (op, name) → np.ndarray (SBERT_DIM,). Experiments with no
    log CSV, or whose log is empty after leakage filtering, fall back to an
    all-zero embedding.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    texts: List[str] = []
    keys: List[Tuple[str, str]] = []
    zero_keys: List[Tuple[str, str]] = []
    n_dropped_total, n_rows_total = 0, 0

    for meta in experiments:
        op, name = meta["op"], meta["name"]
        text, n_rows, n_dropped = load_experiment_text(data_root, system, op, name)
        n_rows_total += n_rows
        n_dropped_total += n_dropped
        if text:
            texts.append(text)
            keys.append((op, name))
        else:
            zero_keys.append((op, name))

    cache: Dict[Tuple[str, str], np.ndarray] = {}
    if texts:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for key, emb in zip(keys, embeddings):
            cache[key] = emb.astype(np.float32)
    for key in zero_keys:
        cache[key] = np.zeros(SBERT_DIM, dtype=np.float32)

    print(f"[Text] Built SBERT cache for {len(cache)} experiments "
          f"({len(zero_keys)} zero-filled — no usable log text).")
    print(f"[Text] Leakage filter dropped {n_dropped_total}/{n_rows_total} "
          f"log rows across all experiments (keyword match).")
    return cache
