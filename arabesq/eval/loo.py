
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arabesq.train.trainer import train_model, TrainConfig


@dataclass
class LOOConfig:
    strict: bool = True
    max_holds: Optional[int] = None


def loo_strict(
    model_ctor,
    base_data,
    y,
    base_train_mask,
    val_mask,
    cfg_train: TrainConfig,
    cfg_loo: LOOConfig,
    out_dir: str,
    logger,
):
    """Strict leave-one-out: retrain a fresh model for each held-out train node.

    Returns:
      preds: (H,3) marginal probs for multilabel; or (H,4) probs for softmax4/dirichlet4
      trues: (H,*) raw y rows (depends on out_mode)
      held_ids: list[int]
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_idx = torch.where(base_train_mask)[0].cpu().numpy()
    if cfg_loo.max_holds is not None:
        train_idx = train_idx[: cfg_loo.max_holds]

    preds, trues, held_ids = [], [], []

    for t, hold in enumerate(train_idx, 1):
        logger.info(f"LOO {t}/{len(train_idx)} holdout node={hold}")
        train_mask = base_train_mask.clone()
        train_mask[hold] = False

        model = model_ctor()
        run_dir = str(Path(out_dir) / f"hold_{hold:06d}")
        ckpt = train_model(model, base_data, y, train_mask, val_mask, cfg_train, run_dir, logger)

        state = torch.load(ckpt, map_location=cfg_train.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        model = model.to(cfg_train.device).eval()
        with torch.no_grad():
            out = model(base_data.to(cfg_train.device))
            if cfg_train.out_mode == "multilabel":
                p = out["probs_marginal3"][hold].detach().cpu().numpy()
            else:
                p = out["probs4"][hold].detach().cpu().numpy()

        preds.append(p)
        trues.append(y[hold].detach().cpu().numpy())
        held_ids.append(int(hold))

    return np.vstack(preds), np.vstack(trues), held_ids
