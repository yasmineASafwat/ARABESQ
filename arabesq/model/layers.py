
from __future__ import annotations

import torch
import torch.nn as nn

class SemanticAttention(nn.Module):
    """Semantic-level attention to fuse relation-specific embeddings.

    Given list of relation embeddings [N, D] each, returns fused [N, D] and weights [R].
    This follows the common 'semantic attention' design used in heterogeneous GNNs (e.g., HAN-style).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse relation embeddings.

        Returns:
          fused: (N,D)
          beta_node: (N,R) node-specific semantic weights ("for a given protein")
          beta_mean: (R,) mean weight across nodes (useful for logging/plots)
        """
        # scores_rn: (R, N, 1)
        scores = torch.stack([self.proj(h) for h in hs], dim=0)
        # node-wise softmax over relations: (R, N)
        beta_rn = torch.softmax(scores.squeeze(-1), dim=0)
        # fuse
        out = torch.zeros_like(hs[0])
        for r, h in enumerate(hs):
            out = out + beta_rn[r].unsqueeze(-1) * h
        beta_node = beta_rn.transpose(0, 1).contiguous()  # (N,R)
        beta_mean = beta_node.mean(dim=0)
        return out, beta_node, beta_mean

class RelationDropout(nn.Module):
    """Randomly drop entire relation embeddings during training."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def forward(self, hs: list[torch.Tensor]) -> list[torch.Tensor]:
        if (not self.training) or self.p <= 0:
            return hs
        keep = torch.rand(len(hs), device=hs[0].device) > self.p
        if keep.sum() == 0:
            keep[torch.randint(0, len(hs), (1,), device=hs[0].device)] = True
        return [h if k else torch.zeros_like(h) for k, h in zip(keep.tolist(), hs)]
