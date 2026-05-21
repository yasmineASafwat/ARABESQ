"""A lightweight GAT-style layer with **edge-weighted attention**.

    This is implemented to match the paper-facing requirement:
      - If an edge confidence s_ij is low, the model should mathematically
        down-weight (and potentially ignore) that neighbor.

    Mechanism:
      - Standard attention logits e_ij are computed from projected node features.
      - We inject the edge confidence s_ij in [0,1] by:
            e'_ij = e_ij + log(s_ij + eps)
        which makes s_ij=0 effectively remove the edge (log(0) -> -inf).
      - Attention coefficients are alpha_ij = softmax_j(e'_ij).

    Notes:
      - For non-confidence edge features (e.g., raw cosine similarity in [-1,1]),
        set use_edge_log_weight=False and optionally pass edge_feat into edge_mlp.
      - This layer is intentionally simple and production-stable; it does not
        depend on private internals of PyG's GATv2Conv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional


class WeightedGATConv(MessagePassing):
    """A lightweight GAT-style layer with **edge-weighted attention**.

    - Inject edge confidence s_ij in [0,1] via:
          e'_ij = e_ij + log(s_ij + eps)
      so very low confidence effectively mutes an edge.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        add_self_loops: bool = False,
        use_edge_log_weight: bool = True,
        edge_feat_dim: Optional[int] = None,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.use_edge_log_weight = use_edge_log_weight

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))

        self.edge_mlp = None
        if edge_feat_dim is not None and edge_feat_dim > 0:
            self.edge_mlp = nn.Sequential(nn.Linear(edge_feat_dim, heads))

        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_mlp is not None:
            for m in self.edge_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (N, in_dim)
        h = self.lin(x).view(-1, self.heads, self.out_dim)  # (N,H,C)
        out = self.propagate(edge_index, x=h, edge_weight=edge_weight, edge_feat=edge_feat)
        out = out.view(-1, self.heads * self.out_dim) + self.bias
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i: int,
        edge_weight: Optional[torch.Tensor],
        edge_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # e_ij per head
        e = (x_i * self.att_dst).sum(dim=-1) + (x_j * self.att_src).sum(dim=-1)  # (E,H)
        e = F.leaky_relu(e, negative_slope=0.2)

        if self.edge_mlp is not None and edge_feat is not None:
            e = e + self.edge_mlp(edge_feat)  # (E,H)

        if self.use_edge_log_weight and edge_weight is not None:
            ew = edge_weight.view(-1, 1).clamp(min=0.0, max=1.0)
            e = e + torch.log(ew + 1e-12)

        alpha = softmax(e, index)  # (E,H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)  # (E,H,C)
