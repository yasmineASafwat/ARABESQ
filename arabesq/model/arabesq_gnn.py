
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from arabesq.model.weighted_gat import WeightedGATConv

from arabesq.model.layers import SemanticAttention, RelationDropout
from arabesq.model.evidential import dirichlet_alpha, dirichlet_mean, dirichlet_uncertainty

@dataclass
class arabesqModelConfig:
    in_dim: int
    hidden_dim: int = 256
    heads: int = 4
    dropout: float = 0.2
    relation_dropout: float = 0.2
    out_mode: str = "multilabel"  # 'multilabel', 'softmax4', or 'dirichlet4'
    coex_gate_hidden: int = 32
    ppi_high_th: float = 0.7
    use_evidence_tiers: bool = True
    bilinear_gate_dim: int = 64

class arabesqGNN(nn.Module):
    """Multi-relational protein graph model with COEX edge gating.

    Relations:
      - sim: KNN on ESM-2 embeddings (edge_attr: cosine similarity scalar)
      - ppi: physical interaction edges (edge_attr: confidence scalar in [0,1])
      - coex: coexpression edges

    COEX Edge Gating:
      For each COEX edge, we attach features:
        [coex_weight, cosine_sim, ppi_flag]
      The model computes a learnable gate g in (0,1) via an MLP:
        g = sigmoid(MLP([coex_weight, cosine_sim, ppi_flag]))
      and passes the **gated weight** (coex_weight * g) into the COEX GAT encoder.

    Fusion:
      - Separate GATv2Conv per relation (edge_attr injected via edge_dim=1).
      - RelationDropout to improve cold-start robustness.
      - SemanticAttention to fuse relation embeddings.

    Output heads:
      - multilabel: 3 logits (mito, plastid, other) with sigmoid.
      - dirichlet4: 4 logits (mito-only, plastid-only, dual, other) with evidential uncertainty.
        Marginal probabilities for threshold logic:
            p_mito = p(mito-only) + p(dual)
            p_plastid = p(plastid-only) + p(dual)
            p_other = p(other)
    """

    def __init__(self, cfg: arabesqModelConfig):
        super().__init__()
        self.cfg = cfg
        out_per_head = cfg.hidden_dim // cfg.heads

        # SIM uses edge features (cosine) but not log-weight scaling (cosine may be negative)
        self.conv_sim = WeightedGATConv(
            in_dim=cfg.in_dim,
            out_dim=out_per_head,
            heads=cfg.heads,
            dropout=cfg.dropout,
            use_edge_log_weight=False,
            edge_feat_dim=1,
        )

        # PPI uses confidence as an edge weight in [0,1] via log-weight scaling
        self.conv_ppi_high = WeightedGATConv(
            in_dim=cfg.in_dim,
            out_dim=out_per_head,
            heads=cfg.heads,
            dropout=cfg.dropout,
            use_edge_log_weight=True,
            edge_feat_dim=None,
        )
        self.conv_ppi_low = WeightedGATConv(
            in_dim=cfg.hidden_dim,
            out_dim=out_per_head,
            heads=cfg.heads,
            dropout=cfg.dropout,
            use_edge_log_weight=True,
            edge_feat_dim=1,  # allows adding an alignment gate
        )

        # COEX consumes a gated scalar weight in [0,1] via log-weight scaling
        self.conv_coex = WeightedGATConv(
            in_dim=cfg.hidden_dim,
            out_dim=out_per_head,
            heads=cfg.heads,
            dropout=cfg.dropout,
            use_edge_log_weight=True,
            edge_feat_dim=None,
        )

        # --- COEX Bilinear Integration gate ---
        # Project node features to a smaller space, then compute bilinear score between endpoints.
        self.coex_proj = nn.Linear(cfg.in_dim, cfg.bilinear_gate_dim)
        self.coex_bilinear = nn.Bilinear(cfg.bilinear_gate_dim, cfg.bilinear_gate_dim, 1, bias=False)
        # Combine bilinear score with edge features [coex_weight, cosine_sim, ppi_flag]
        self.coex_edge_lin = nn.Linear(3, 1)

        self.rel_drop = RelationDropout(cfg.relation_dropout)
        self.sem_attn = SemanticAttention(cfg.hidden_dim, hidden_dim=128)

        out_dim = 3 if cfg.out_mode == "multilabel" else 4
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x = data["protein"].x

        # -------- Tier 1: Sequence similarity (always present) --------
        et_sim = ("protein", "sim", "protein")
        h_sim = self.conv_sim(
            x,
            data[et_sim].edge_index,
            edge_weight=None,
            edge_feat=data[et_sim].edge_attr,
        )
        h_sim = F.elu(h_sim)

        # -------- Tier 2: High-confidence PPI (strong external evidence) --------
        h_ppi_high = torch.zeros(x.size(0), self.cfg.hidden_dim, device=x.device)
        ppi_edge_index_high = None
        ppi_conf_high = None

        if ("protein", "ppi", "protein") in data.edge_types:
            et_ppi = ("protein", "ppi", "protein")
            conf = data[et_ppi].edge_attr.view(-1)
            mask_high = conf >= self.cfg.ppi_high_th
            if mask_high.any():
                ppi_edge_index_high = data[et_ppi].edge_index[:, mask_high]
                ppi_conf_high = conf[mask_high].view(-1, 1)
                h_ppi_high = self.conv_ppi_high(
                    x,
                    ppi_edge_index_high,
                    edge_weight=ppi_conf_high,
                    edge_feat=None,
                )
                h_ppi_high = F.elu(h_ppi_high)

        # Fuse Tier1 + Tier2 (node-wise semantic attention); can favor PPI when present.
        hs12 = [h_sim, h_ppi_high]
        hs12 = self.rel_drop(hs12)
        h12, beta12_node, beta12_mean = self.sem_attn(hs12)

        # -------- Tier 3: Contextual evidence (COEX + Low-conf PPI), gated by alignment --------
        h_tier3 = torch.zeros_like(h12)
        beta3_mean = torch.zeros(2, device=x.device)  # placeholder logging [ppi_low, coex]

        if self.cfg.use_evidence_tiers:
            # Low-confidence PPI gated by alignment (edge_feat = alignment score)
            h_ppi_low = torch.zeros_like(h12)
            if ("protein", "ppi", "protein") in data.edge_types:
                et_ppi = ("protein", "ppi", "protein")
                conf = data[et_ppi].edge_attr.view(-1)
                mask_low = conf < self.cfg.ppi_high_th
                if mask_low.any():
                    ei_low = data[et_ppi].edge_index[:, mask_low]
                    conf_low = conf[mask_low].view(-1, 1)
                    # alignment = cosine between tier12 embeddings
                    src, dst = ei_low[0], ei_low[1]
                    h_src = F.normalize(h12[src], dim=-1)
                    h_dst = F.normalize(h12[dst], dim=-1)
                    align = (h_src * h_dst).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                    # map to [0,1] for numerical stability
                    align01 = (align + 1.0) / 2.0
                    conf_eff = (conf_low * align01).clamp(0.0, 1.0)
                    h_ppi_low = self.conv_ppi_low(
                        h12,
                        ei_low,
                        edge_weight=conf_eff,
                        edge_feat=align01,
                    )
                    h_ppi_low = F.elu(h_ppi_low)

            # COEX bilinear integration gate
            h_coex = torch.zeros_like(h12)
            if ("protein", "coex", "protein") in data.edge_types:
                et_cx = ("protein", "coex", "protein")
                ei = data[et_cx].edge_index
                ef = data[et_cx].edge_attr  # (E,3) = [coex_w, sim, ppi_flag]
                # bilinear score between endpoints in projected space
                z = self.coex_proj(x)  # use Tier1 features as "functional profile" anchor
                src, dst = ei[0], ei[1]
                bil = self.coex_bilinear(z[src], z[dst])  # (E,1)
                gate_logits = bil + self.coex_edge_lin(ef)
                gate = torch.sigmoid(gate_logits)
                coex_eff = (ef[:, 0:1] * gate).clamp(0.0, 1.0)
                h_coex = self.conv_coex(
                    h12,
                    ei,
                    edge_weight=coex_eff,
                    edge_feat=None,
                )
                h_coex = F.elu(h_coex)

            # Fuse tier3 sources (node-wise attention)
            hs3 = [h_ppi_low, h_coex]
            hs3 = self.rel_drop(hs3)
            h_tier3, beta3_node, beta3_mean = self.sem_attn(hs3)

        # Final embedding
        h_final = h12 + h_tier3
        logits = self.mlp(h_final)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "semantic_weights_tier12_node": beta12_node,
            "semantic_weights_tier12_mean": beta12_mean,
            "semantic_weights_tier3_mean": beta3_mean,
        }

        if self.cfg.out_mode == "multilabel":
            out["probs_marginal3"] = torch.sigmoid(logits)
        elif self.cfg.out_mode == "softmax4":
            p4 = torch.softmax(logits, dim=-1)
            out["probs4"] = p4
            # Also export 3 marginals for optional analysis
            p_mito = p4[:, 0]
            p_plastid = p4[:, 1]
            p_dual = p4[:, 2]
            p_other = p4[:, 3]
            out["probs_marginal3"] = torch.stack([p_mito + p_dual, p_plastid + p_dual, p_other], dim=-1)
        elif self.cfg.out_mode == "dirichlet4":
            alpha = dirichlet_alpha(logits)
            p4 = dirichlet_mean(alpha)
            out["alpha"] = alpha
            out["probs4"] = p4
            out["uncertainty"] = dirichlet_uncertainty(alpha)
            p_mito = p4[:, 0] + p4[:, 2]
            p_plastid = p4[:, 1] + p4[:, 2]
            p_other = p4[:, 3]
            out["probs_marginal3"] = torch.stack([p_mito, p_plastid, p_other], dim=-1)
        else:
            raise ValueError(f"Unknown out_mode: {self.cfg.out_mode}")

        return out
