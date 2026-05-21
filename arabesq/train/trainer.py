from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import os
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from arabesq.model.evidential import dirichlet_alpha, edl_mse_loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_mode: str = "multilabel"  # must match model cfg
    edl_kl: float = 1e-3


def _bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


def class_weights_from_train(y_long: torch.Tensor, train_mask: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency class weights computed on the training subset."""
    counts = torch.bincount(y_long[train_mask], minlength=num_classes).float()
    w = counts.sum() / (num_classes * (counts + 1e-12))
    return w


def _to_plain(obj: Any) -> Any:
    """
    Convert config-like objects (dataclasses / simple classes / PathLike) into
    plain Python containers (dict/list/str/int/float/bool/None).
    This keeps checkpoints compatible with torch.load(weights_only=True).
    """
    if obj is None:
        return None

    # dataclass -> dict
    try:
        if hasattr(obj, "__dataclass_fields__"):
            return _to_plain(asdict(obj))
    except Exception:
        pass

    # common "to_dict"
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_plain(obj.to_dict())
        except Exception:
            pass

    # plain dict
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    # primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Path-like
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)

    # fallback: object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _to_plain(dict(obj.__dict__))
        except Exception:
            pass

    # last resort: string representation
    return str(obj)


def train_model(
    model: torch.nn.Module,
    data: HeteroData,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: TrainConfig,
    out_dir: str,
    logger,
) -> str:
    """Train transductively on a single HeteroData graph, using masks.

    Targets:
      - multilabel: y is (N,3) with multi-hot labels (mito, plastid, other)
      - softmax4:   y is (N,) long in {0,1,2,3} for (mito-only, plastid-only, dual, other)
      - dirichlet4: y is (N,4) one-hot (mito-only, plastid-only, dual, other)
    """
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(cfg.device)
    data = data.to(cfg.device)
    y = y.to(cfg.device)
    train_mask = train_mask.to(cfg.device)
    val_mask = val_mask.to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    best_epoch = 0
    best_path = str(Path(out_dir) / "model_best.pt")
    bad = 0

    # Capture model config in a safe way (plain dict / primitives only)
    model_cfg_obj = getattr(model, "cfg", None)
    model_cfg_plain = _to_plain(model_cfg_obj)  # <- crucial change
    train_cfg_plain = _to_plain(cfg)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data)
        logits = out["logits"]

        if cfg.out_mode == "multilabel":
            loss = _bce_loss(logits[train_mask], y[train_mask])
        elif cfg.out_mode == "softmax4":
            w = class_weights_from_train(y, train_mask, num_classes=4).to(cfg.device)
            loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=w)
        elif cfg.out_mode == "dirichlet4":
            alpha = dirichlet_alpha(logits[train_mask])
            loss = edl_mse_loss(alpha, y[train_mask], coeff_kl=cfg.edl_kl)
        else:
            raise ValueError(cfg.out_mode)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outv = model(data)
            logitsv = outv["logits"]
            if cfg.out_mode == "multilabel":
                vloss = _bce_loss(logitsv[val_mask], y[val_mask]).item()
            elif cfg.out_mode == "softmax4":
                w = class_weights_from_train(y, train_mask, num_classes=4).to(cfg.device)
                vloss = F.cross_entropy(logitsv[val_mask], y[val_mask], weight=w).item()
            else:
                alpha = dirichlet_alpha(logitsv[val_mask])
                vloss = edl_mse_loss(alpha, y[val_mask], coeff_kl=cfg.edl_kl).item()

        logger.info(f"epoch={epoch:03d} train_loss={loss.item():.4f} val_loss={vloss:.4f}")

        if vloss < best - 1e-5:
            best = vloss
            best_epoch = epoch
            bad = 0

            # Save ONLY tensors + plain dicts (no custom classes)
            ckpt = {
                "model_state": model.state_dict(),
                # Keep legacy key name "cfg" for compatibility, but store dict not object:
                "cfg": model_cfg_plain,
                # Also store clearer names:
                "model_cfg": model_cfg_plain,
                "train_cfg": train_cfg_plain,
                "best_val_loss": float(best),
                "best_epoch": int(best_epoch),
            }
            torch.save(ckpt, best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                logger.info(f"Early stopping at epoch={epoch} (best val loss={best:.4f} at epoch={best_epoch}).")
                break

    return best_path