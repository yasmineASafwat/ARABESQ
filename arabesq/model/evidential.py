
from __future__ import annotations

import torch
import torch.nn.functional as F

def evidence_softplus(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits)

def dirichlet_alpha(logits: torch.Tensor) -> torch.Tensor:
    e = evidence_softplus(logits)
    return e + 1.0

def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True)

def dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    # u = K / S where S=sum(alpha), K=number of classes
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1)
    return K / (S + 1e-12)

def edl_mse_loss(alpha: torch.Tensor, y_onehot: torch.Tensor, coeff_kl: float = 1e-3) -> torch.Tensor:
    """Simplified evidential loss: MSE on Dirichlet mean + KL-to-uniform regularizer."""
    p = dirichlet_mean(alpha)
    mse = ((y_onehot - p) ** 2).sum(dim=-1).mean()

    K = alpha.shape[-1]
    alpha0 = alpha.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(alpha0) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        + torch.lgamma(torch.tensor(float(K), device=alpha.device)) - K * torch.lgamma(torch.tensor(1.0, device=alpha.device))
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0))).sum(dim=-1, keepdim=True)
    ).mean()
    return mse + coeff_kl * kl
