from __future__ import annotations

"""SUBA/STRING edge readers + weight derivations.

This module is intentionally conservative and explicit about semantics.

Key user-provided SUBA COEX semantics:
  - new_rank (Mutual Rank, MR): 1..100, **lower is stronger**.
  - ave: ~1..25, **higher is stronger**.
  - connect: {match, adjacent, distant, unclear} is a *spatial* category.
    For localization, 'distant'/'unclear' are treated as noise.

We turn these into a single `coex_weight` in [0,1] with default:
  coex_weight = combine(mr_norm, ave_norm) * connect_factor
where combine defaults to geometric mean.

For PPI, confidence is derived either from a numeric score column (STRING, 0..1)
or paper counts (SUBA), and then multiplied by an optional connect_factor.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from arabesq.utils.ids import normalize_gene_id


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _canonicalize_pairs(df: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    df = df.copy()
    df[a_col] = df[a_col].map(normalize_gene_id)
    df[b_col] = df[b_col].map(normalize_gene_id)
    canon = df.apply(lambda r: _canon_pair(r[a_col], r[b_col]), axis=1, result_type="expand")
    df[a_col], df[b_col] = canon[0], canon[1]
    return df


def _connect_factor(series: pd.Series, connect_weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Map SUBA spatial categories to [0,1] weights.

    Defaults (as requested):
      match=1.0, adjacent=0.5, distant=0.0, unclear=0.0
    """
    cw = connect_weights or {"MATCH": 1.0, "ADJACENT": 0.5, "DISTANT": 0.0, "UNCLEAR": 0.0}
    return series.astype(str).str.upper().map(cw).fillna(cw.get("UNCLEAR", 0.0)).astype(float)


def _find_numeric_like(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Heuristic: choose a candidate column that converts to numeric for most rows."""
    best = None
    best_valid = 0.0
    for c in candidates:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        valid = float(s.notna().mean())
        if valid > best_valid:
            best_valid = valid
            best = c
    return best


def read_ppi(
    ppi_csv: str,
    locusA_col: str = "locusA",
    locusB_col: str = "locusB",
    connect_col: Optional[str] = None,
) -> pd.DataFrame:
    """Read a PPI edge list.

    Required columns: locusA_col, locusB_col.
    Optional columns:
      - paper (SUBA-style)
      - numeric confidence score (STRING-style; use via `conf_col` in derive_ppi_confidence)
      - connect (SUBA spatial category) via connect_col
    """
    df = pd.read_csv(ppi_csv)
    for c in [locusA_col, locusB_col]:
        if c not in df.columns:
            raise ValueError(f"PPI file missing required column '{c}'. Found: {list(df.columns)}")

    df = df.rename(columns={locusA_col: "locusA", locusB_col: "locusB"}).copy()
    if "paper" not in df.columns:
        df["paper"] = "unknown"
    if connect_col is not None and connect_col in df.columns:
        df = df.rename(columns={connect_col: "connect"}).copy()
    if "connect" not in df.columns:
        df["connect"] = "unclear"

    df = _canonicalize_pairs(df, "locusA", "locusB")
    return df


def derive_ppi_confidence(
    df_ppi: pd.DataFrame,
    method: str = "paper_count",
    conf_col: Optional[str] = None,
    connect_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Return undirected PPI pairs with `ppi_conf` in [0,1].

    Priority:
      1) If conf_col exists, use it directly (STRING; clipped to [0,1]).
      2) Else if method == 'paper_count', use publication counts (SUBA):
            ppi_conf_raw = 1 - exp(-n_papers)
      3) Else method == 'binary': ppi_conf_raw=1.0

    If a `connect` column exists, multiply by connect_factor (spatial category weight).
    """
    df = df_ppi.copy()
    if "connect" in df.columns:
        df["connect_factor"] = _connect_factor(df["connect"], connect_weights)
    else:
        df["connect_factor"] = 1.0

    # Case 1: direct confidence
    if conf_col is not None and conf_col in df.columns:
        pairs = df[["locusA", "locusB", conf_col, "connect_factor"]].copy()
        pairs = pairs.rename(columns={conf_col: "ppi_conf_raw"})
        pairs["ppi_conf_raw"] = pd.to_numeric(pairs["ppi_conf_raw"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        pairs = pairs.groupby(["locusA", "locusB"], as_index=False).agg(
            ppi_conf_raw=("ppi_conf_raw", "max"),
            connect_factor=("connect_factor", "max"),
        )
        pairs["ppi_conf"] = (pairs["ppi_conf_raw"] * pairs["connect_factor"]).clip(0.0, 1.0)
        return pairs[["locusA", "locusB", "ppi_conf"]]

    # Case 2/3: derived
    pairs = df.groupby(["locusA", "locusB"], as_index=False).agg(
        n_papers=("paper", "count"),
        connect_factor=("connect_factor", "max"),
    )

    if method == "paper_count":
        pairs["ppi_conf_raw"] = 1.0 - np.exp(-pairs["n_papers"].astype(float))
    elif method == "binary":
        pairs["ppi_conf_raw"] = 1.0
    else:
        raise ValueError(f"Unknown confidence method: {method}")

    pairs["ppi_conf"] = (pairs["ppi_conf_raw"].astype(float) * pairs["connect_factor"].astype(float)).clip(0.0, 1.0)
    return pairs[["locusA", "locusB", "ppi_conf"]]


def read_coex(
    coex_csv: str,
    locusA_col: str = "locusA",
    locusB_col: str = "locusB",
    rank_col: str = "new_rank",
    ave_col: str = "ave",
    connect_col: str = "connect",
) -> pd.DataFrame:
    """Read a COEX edge list.

    Required columns: locusA_col, locusB_col.
    Expected (SUBA):
      - new_rank (MR)
      - ave
      - connect
    """
    df = pd.read_csv(coex_csv)
    for c in [locusA_col, locusB_col]:
        if c not in df.columns:
            raise ValueError(f"COEX file missing required column '{c}'. Found: {list(df.columns)}")

    df = df.rename(columns={locusA_col: "locusA", locusB_col: "locusB"}).copy()

    # MR
    if rank_col in df.columns:
        df = df.rename(columns={rank_col: "new_rank"}).copy()
    if "new_rank" not in df.columns:
        df["new_rank"] = np.nan

    # ave
    if ave_col in df.columns:
        df = df.rename(columns={ave_col: "ave"}).copy()
    if "ave" not in df.columns:
        cand = [c for c in df.columns if "ave" in str(c).lower()]
        pick = _find_numeric_like(df, cand)
        if pick is not None:
            df = df.rename(columns={pick: "ave"}).copy()
        else:
            df["ave"] = np.nan

    # connect
    if connect_col in df.columns:
        df = df.rename(columns={connect_col: "connect"}).copy()
    if "connect" not in df.columns:
        df["connect"] = "unclear"

    df = _canonicalize_pairs(df, "locusA", "locusB")
    return df


def coex_weight(
    df_coex: pd.DataFrame,
    rank_mode: str = "lower_is_stronger",
    weight_col: Optional[str] = None,
    connect_weights: Optional[Dict[str, float]] = None,
    combine: str = "geom",
) -> pd.DataFrame:
    """Return undirected COEX pairs with `coex_weight` in [0,1].

    Priority:
      1) If weight_col exists, use it directly (clipped to [0,1]).
      2) Else derive from SUBA MR (new_rank), ave, and connect.

    Normalizations:
      - MR: lower better → mr_norm in [0,1]
      - ave: higher better → ave_norm in [0,1]

    combine:
      - 'geom' (default): sqrt(mr_norm * ave_norm)
      - 'mean': 0.5*(mr_norm + ave_norm)
      - 'product': mr_norm * ave_norm
    """
    df = df_coex.copy()

    if weight_col is not None and weight_col in df.columns:
        out = df[["locusA", "locusB", weight_col]].copy()
        out = out.rename(columns={weight_col: "coex_weight"})
        out["coex_weight"] = pd.to_numeric(out["coex_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out = out.groupby(["locusA", "locusB"], as_index=False)["coex_weight"].max()
        return out

    # Spatial modifier
    if "connect" in df.columns:
        df["connect_factor"] = _connect_factor(df["connect"], connect_weights)
    else:
        df["connect_factor"] = 1.0

    # MR normalization
    r = pd.to_numeric(df.get("new_rank", np.nan), errors="coerce")
    rmax = float(r.max()) if pd.notnull(r.max()) else 1.0
    if rank_mode == "lower_is_stronger":
        denom = (rmax - 1.0) if rmax > 1.0 else 1.0
        mr_norm = 1.0 - ((r - 1.0) / denom)
    elif rank_mode == "higher_is_stronger":
        denom = rmax if rmax > 0 else 1.0
        mr_norm = r / denom
    elif rank_mode == "none":
        mr_norm = 1.0
    else:
        raise ValueError(f"Unknown rank_mode: {rank_mode}")
    mr_norm = pd.to_numeric(mr_norm, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # ave normalization
    a = pd.to_numeric(df.get("ave", np.nan), errors="coerce")
    amax = float(a.max()) if pd.notnull(a.max()) else 1.0
    denom_a = (amax - 1.0) if amax > 1.0 else 1.0
    ave_norm = (a - 1.0) / denom_a
    ave_norm = pd.to_numeric(ave_norm, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    mr = mr_norm.to_numpy(dtype=float)
    av = ave_norm.to_numpy(dtype=float)
    if combine == "geom":
        base = np.sqrt(mr * av)
    elif combine == "mean":
        base = 0.5 * (mr + av)
    elif combine == "product":
        base = mr * av
    else:
        raise ValueError(f"Unknown combine mode: {combine}")

    w = (base * df["connect_factor"].to_numpy(dtype=float)).clip(0.0, 1.0)
    df["coex_weight"] = w

    out = df[["locusA", "locusB", "coex_weight"]].copy()
    out = out.groupby(["locusA", "locusB"], as_index=False)["coex_weight"].max()
    return out
