
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sklearn.model_selection import train_test_split

from arabesq.utils.ids import normalize_gene_id

REQUIRED_MIN_COLS = ["Gene", "localization", "sequence"]

@dataclass
class DatasetTables:
    train: pd.DataFrame
    val: Optional[pd.DataFrame]
    test: Optional[pd.DataFrame]

def _validate_min_cols(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_MIN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found columns: {list(df.columns)}")

def read_split_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_min_cols(df, path)
    df = df.copy()
    df["Gene_raw"] = df["Gene"]
    df["Gene"] = df["Gene"].map(normalize_gene_id)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    # Normalize localization labels to canonical 4-class strings
    _loc_map = {
        'mito': 'mito',
        'mitochondrion': 'mito',
        'mitochondria': 'mito',
        'plastid': 'plastid',
        'chloroplast': 'plastid',
        'chloro': 'plastid',
        'dual': 'dual',
        'dual-targeted': 'dual',
        'dual_targeted': 'dual',
        'other': 'other',
    }
    df['localization'] = (
        df['localization'].astype(str).str.strip().str.lower()
        .map(_loc_map)
        .fillna(df['localization'].astype(str).str.strip().str.lower())
    )
    return df

def build_tables(train_csv: str, val_csv: Optional[str], test_csv: Optional[str]) -> DatasetTables:
    train = read_split_csv(train_csv)
    val = read_split_csv(val_csv) if val_csv else None
    test = read_split_csv(test_csv) if test_csv else None
    return DatasetTables(train=train, val=val, test=test)


def split_train_val(
    train_df: pd.DataFrame,
    val_frac: float = 0.1,
    mode: str = "auto",
    seed: int = 1337,
    group_col: str = "cluster",
    stratify_col: str = "localization",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a validation split from train.

    Modes:
      - 'stratified': row-level stratified split by `stratify_col`.
      - 'cluster': group split by `group_col` (no stratification).
      - 'cluster_stratified': group split aiming to preserve label proportions.
      - 'auto': if group_col exists -> cluster_stratified else stratified.

    Notes:
      - This function assumes Gene IDs are already normalized (read_split_csv does that).
      - cluster-aware splitting prevents leakage when clusters encode homology or curation.
    """
    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"val_frac must be in (0,0.5). Got {val_frac}")

    df = train_df.copy().reset_index(drop=True)

    if mode == "auto":
        mode = "cluster_stratified" if group_col in df.columns else "stratified"

    if mode == "stratified":
        y = df[stratify_col].astype(str).to_numpy()
        tr, va = train_test_split(df, test_size=val_frac, random_state=seed, stratify=y)
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    if group_col not in df.columns:
        # Fallback to stratified if no groups exist
        y = df[stratify_col].astype(str).to_numpy()
        tr, va = train_test_split(df, test_size=val_frac, random_state=seed, stratify=y)
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    groups = df[group_col].astype(str).fillna("NA").to_numpy()
    y = df[stratify_col].astype(str).to_numpy()

    # ---- cluster only: random group holdout ----
    if mode == "cluster":
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        target_n = int(round(val_frac * len(df)))
        val_groups = set()
        cur = 0
        for g in uniq:
            idx = np.where(groups == g)[0]
            if cur < target_n:
                val_groups.add(g)
                cur += len(idx)
        is_val = np.array([g in val_groups for g in groups], dtype=bool)
        va = df[is_val]
        tr = df[~is_val]
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    # ---- cluster_stratified: greedy group assignment to match label proportions ----
    if mode != "cluster_stratified":
        raise ValueError(f"Unknown split mode: {mode}")

    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    label_to_i = {lab: i for i, lab in enumerate(labels)}

    # Per-group label count vectors
    uniq_groups = np.unique(groups)
    group_counts = {}
    group_sizes = {}
    for g in uniq_groups:
        idx = np.where(groups == g)[0]
        group_sizes[g] = len(idx)
        cnt = np.zeros(len(labels), dtype=float)
        for lab in y[idx]:
            cnt[label_to_i[lab]] += 1.0
        group_counts[g] = cnt

    total = np.zeros(len(labels), dtype=float)
    for lab in y:
        total[label_to_i[lab]] += 1.0
    target_prop = total / max(total.sum(), 1.0)
    target_n = int(round(val_frac * len(df)))

    # Stable ordering: larger groups first, with small random tie-break
    order = list(uniq_groups)
    rng.shuffle(order)
    order.sort(key=lambda g: (-group_sizes[g], str(g)))

    val_groups = set()
    val_cnt = np.zeros(len(labels), dtype=float)
    val_size = 0

    def cost(next_cnt: np.ndarray, next_size: int) -> float:
        prop = next_cnt / max(next_cnt.sum(), 1.0)
        # L1 distance to target label proportions + size penalty
        prop_cost = float(np.abs(prop - target_prop).sum())
        size_cost = abs(next_size - target_n) / max(target_n, 1)
        return prop_cost + 0.25 * size_cost

    remaining = set(order)
    # Greedily add groups until reaching target size, minimizing cost
    while val_size < target_n and remaining:
        best_g = None
        best_c = None
        for g in list(remaining):
            ncnt = val_cnt + group_counts[g]
            nsz = val_size + group_sizes[g]
            c = cost(ncnt, nsz)
            if best_c is None or c < best_c:
                best_c = c
                best_g = g
        val_groups.add(best_g)
        val_cnt += group_counts[best_g]
        val_size += group_sizes[best_g]
        remaining.remove(best_g)

    is_val = np.array([g in val_groups for g in groups], dtype=bool)
    va = df[is_val]
    tr = df[~is_val]
    return tr.reset_index(drop=True), va.reset_index(drop=True)
