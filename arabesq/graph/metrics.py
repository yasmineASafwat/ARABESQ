
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData

@dataclass
class GraphStats:
    n_nodes: int
    n_edges_total: int
    n_edges_by_type: Dict[str,int]
    density_union: float
    n_components_union: int
    avg_degree_union: float
    clustering_coeff_union: float

def _to_nx_undirected(edge_index: np.ndarray, n_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return g

def compute_graph_stats(data: HeteroData) -> GraphStats:
    n = data["protein"].num_nodes
    n_edges_by_type = {}
    union_edges = []
    for et in data.edge_types:
        ei = data[et].edge_index.cpu().numpy()
        n_edges_by_type[str(et)] = int(ei.shape[1])
        union_edges.append(ei)
    n_total = int(sum(n_edges_by_type.values()))
    if len(union_edges) == 0:
        g_union = nx.Graph()
        g_union.add_nodes_from(range(n))
    else:
        ei_union = np.concatenate(union_edges, axis=1)
        g_union = _to_nx_undirected(ei_union, n)

    # density in undirected graph
    density = nx.density(g_union) if n > 1 else 0.0
    comps = nx.number_connected_components(g_union)
    avg_deg = float(np.mean([d for _, d in g_union.degree()])) if n > 0 else 0.0
    clustering = nx.average_clustering(g_union) if n > 2 else 0.0

    return GraphStats(
        n_nodes=int(n),
        n_edges_total=n_total,
        n_edges_by_type=n_edges_by_type,
        density_union=float(density),
        n_components_union=int(comps),
        avg_degree_union=float(avg_deg),
        clustering_coeff_union=float(clustering),
    )

def to_dict(stats: GraphStats) -> Dict:
    return asdict(stats)
