
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from arabesq.utils.ids import normalize_gene_id

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T

def knn_edges(X: np.ndarray, k: int = 20, self_loops: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edge_index, edge_weight) for a directed KNN graph using cosine similarity."""
    sim = cosine_sim_matrix(X)
    n = sim.shape[0]
    k_eff = min(k + 1, n)
    idx = np.argpartition(-sim, kth=np.arange(k_eff), axis=1)[:, :k_eff]

    rows = np.repeat(np.arange(n), k_eff)
    cols = idx.reshape(-1)
    w = sim[rows, cols]
    if not self_loops:
        mask = rows != cols
        rows, cols, w = rows[mask], cols[mask], w[mask]
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = w.astype(np.float32)
    return edge_index, edge_weight

def _pairs_to_edge_index(
    pairs: pd.DataFrame,
    nodes_map: Dict[str, int],
    a: str,
    b: str,
    make_undirected: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe of pairs into edge_index and the used row indices."""
    src, dst, used = [], [], []
    for ridx, (la, lb) in enumerate(zip(pairs[a].tolist(), pairs[b].tolist())):
        if la in nodes_map and lb in nodes_map:
            src.append(nodes_map[la]); dst.append(nodes_map[lb]); used.append(ridx)
            if make_undirected and la != lb:
                src.append(nodes_map[lb]); dst.append(nodes_map[la]); used.append(ridx)
    edge_index = np.array([src, dst], dtype=np.int64)
    used_idx = np.array(used, dtype=np.int64)
    return edge_index, used_idx

def build_hetero_graph(
    proteins: List[str],
    X: np.ndarray,
    edges_sim: Tuple[np.ndarray, np.ndarray],
    ppi_pairs: Optional[pd.DataFrame] = None,
    coex_pairs: Optional[pd.DataFrame] = None,
    coex_gate_baseline: bool = True,
    gate_tau_sim: float = 0.25,
) -> HeteroData:
    """Build HeteroData with node type 'protein' and relation types sim/ppi/coex.

    COEX edge gating:
      - This function always attaches COEX edge features:
            [coex_weight, cosine_sim, ppi_flag]
      - If `coex_gate_baseline` is True, we *also* apply a conservative deterministic mute:
            if (ppi_flag==0) AND (cosine_sim < gate_tau_sim) => coex_weight := 0
        This is a safe default to prevent obvious false positives.
      - The ARABESQ model additionally has a *learnable* gate that uses these 3 features.
    """
    data = HeteroData()
    data["protein"].x = torch.tensor(X, dtype=torch.float32)
    data["protein"].protein_id = proteins  # metadata

    nodes_map = {p: i for i, p in enumerate(proteins)}

    # SIM
    sim_ei, sim_w = edges_sim
    data["protein", "sim", "protein"].edge_index = torch.tensor(sim_ei, dtype=torch.long)
    data["protein", "sim", "protein"].edge_attr = torch.tensor(sim_w.reshape(-1, 1), dtype=torch.float32)

    # PPI
    ppi_set = set()
    if ppi_pairs is not None and len(ppi_pairs) > 0:
        ppi_pairs = ppi_pairs.copy()
        ppi_pairs["locusA"] = ppi_pairs["locusA"].map(normalize_gene_id)
        ppi_pairs["locusB"] = ppi_pairs["locusB"].map(normalize_gene_id)
        ei, used = _pairs_to_edge_index(ppi_pairs, nodes_map, "locusA", "locusB", make_undirected=True)
        if ei.shape[1] > 0:
            conf = ppi_pairs.iloc[used]["ppi_conf"].to_numpy(dtype=np.float32).reshape(-1, 1)
            data["protein", "ppi", "protein"].edge_index = torch.tensor(ei, dtype=torch.long)
            data["protein", "ppi", "protein"].edge_attr = torch.tensor(conf, dtype=torch.float32)
            ppi_set = set(zip(ei[0].tolist(), ei[1].tolist()))

    # COEX
    if coex_pairs is not None and len(coex_pairs) > 0:
        coex_pairs = coex_pairs.copy()
        coex_pairs["locusA"] = coex_pairs["locusA"].map(normalize_gene_id)
        coex_pairs["locusB"] = coex_pairs["locusB"].map(normalize_gene_id)
        ei, used = _pairs_to_edge_index(coex_pairs, nodes_map, "locusA", "locusB", make_undirected=True)
        if ei.shape[1] > 0:
            w = coex_pairs.iloc[used]["coex_weight"].to_numpy(dtype=np.float32)

            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            src = ei[0]; dst = ei[1]
            sim_edge = (Xn[src] * Xn[dst]).sum(axis=1).astype(np.float32)

            ppi_flag = np.array([(s, d) in ppi_set for s, d in zip(src, dst)], dtype=np.float32)

            if coex_gate_baseline:
                mute = (ppi_flag < 0.5) & (sim_edge < gate_tau_sim)
                w = w.copy()
                w[mute] = 0.0

            edge_feat = np.stack([w, sim_edge, ppi_flag], axis=1).astype(np.float32)  # (E,3)
            data["protein", "coex", "protein"].edge_index = torch.tensor(ei, dtype=torch.long)
            data["protein", "coex", "protein"].edge_attr = torch.tensor(edge_feat, dtype=torch.float32)

    return data
