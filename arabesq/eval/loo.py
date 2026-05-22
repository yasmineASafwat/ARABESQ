from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gc
import shutil

import numpy as np
import torch

from arabesq.train.trainer import train_model, TrainConfig


@dataclass
class LOOConfig:
    strict: bool = True
    max_holds: Optional[int] = None

    # Disk/memory safety options.
    # keep_hold_dirs=False deletes each hold_XXXXXX folder after its prediction is saved.
    # This prevents hundreds of checkpoints/log folders from filling HPC home/scratch quota.
    keep_hold_dirs: bool = False

    # resume=True lets a crashed/killed LOO run continue from loo_partial_predictions.npz.
    resume: bool = True

    # Save partial predictions every N holdouts.
    save_every: int = 1


def _cleanup_after_hold(model=None, state=None, out=None, train_mask=None, run_dir: Optional[Path] = None,
                        keep_hold_dirs: bool = False, logger=None):
    """Release Python/GPU memory and optionally remove per-holdout output directory."""
    try:
        del model
    except Exception:
        pass
    try:
        del state
    except Exception:
        pass
    try:
        del out
    except Exception:
        pass
    try:
        del train_mask
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if run_dir is not None and not keep_hold_dirs:
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Could not remove LOO holdout directory {run_dir}: {exc}")


def _save_partial(partial_path: Path, preds, trues, held_ids):
    """Save partial LOO outputs so long runs can be resumed after kill/quota errors."""
    if len(held_ids) == 0:
        return

    np.savez_compressed(
        partial_path,
        preds=np.vstack(preds),
        trues=np.vstack(trues),
        held_ids=np.asarray(held_ids, dtype=int),
    )


def _load_partial(partial_path: Path, logger=None):
    """Load partial LOO outputs if present."""
    if not partial_path.exists():
        return [], [], []

    arr = np.load(partial_path, allow_pickle=False)
    preds = [x for x in arr["preds"]]
    trues = [x for x in arr["trues"]]
    held_ids = [int(x) for x in arr["held_ids"].tolist()]

    if logger is not None:
        logger.info(f"Resuming LOO from {partial_path}: {len(held_ids)} completed holdouts.")

    return preds, trues, held_ids


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

    Important implementation details:
      - A fresh model is trained for each held-out node.
      - Each holdout directory is deleted by default after prediction is extracted.
      - Partial predictions are saved after each holdout, allowing resume after SLURM kill.
      - CUDA cache is cleared after each holdout to reduce GPU memory accumulation.

    Returns:
      preds: (H,3) marginal probs for multilabel; or (H,4) probs for softmax4/dirichlet4
      trues: (H,*) raw y rows (depends on out_mode)
      held_ids: list[int]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partial_path = out_dir / "loo_partial_predictions.npz"

    train_idx = torch.where(base_train_mask)[0].cpu().numpy()
    if cfg_loo.max_holds is not None:
        train_idx = train_idx[: cfg_loo.max_holds]

    if cfg_loo.resume:
        preds, trues, held_ids = _load_partial(partial_path, logger=logger)
    else:
        preds, trues, held_ids = [], [], []

    completed = set(held_ids)

    # Move base graph to the requested device once. This avoids repeated .to(device)
    # calls that can contribute to memory fragmentation in long LOO runs.
    base_data = base_data.to(cfg_train.device)
    y = y.to(cfg_train.device)
    base_train_mask = base_train_mask.to(cfg_train.device)
    val_mask = val_mask.to(cfg_train.device)

    total = len(train_idx)

    for t, hold_np in enumerate(train_idx, 1):
        hold = int(hold_np)

        if hold in completed:
            logger.info(f"LOO {t}/{total} holdout node={hold} already completed; skipping.")
            continue

        logger.info(f"LOO {t}/{total} holdout node={hold}")

        model = None
        state = None
        out = None
        train_mask = None
        run_dir = out_dir / f"hold_{hold:06d}"

        try:
            train_mask = base_train_mask.clone()
            train_mask[hold] = False

            model = model_ctor()
            ckpt = train_model(
                model,
                base_data,
                y,
                train_mask,
                val_mask,
                cfg_train,
                str(run_dir),
                logger,
            )

            state = torch.load(ckpt, map_location=cfg_train.device, weights_only=True)
            model.load_state_dict(state["model_state"])
            model = model.to(cfg_train.device).eval()

            with torch.no_grad():
                out = model(base_data)
                if cfg_train.out_mode == "multilabel":
                    p = out["probs_marginal3"][hold].detach().cpu().numpy()
                else:
                    p = out["probs4"][hold].detach().cpu().numpy()

            preds.append(p)
            trues.append(y[hold].detach().cpu().numpy())
            held_ids.append(hold)
            completed.add(hold)

            if cfg_loo.save_every <= 1 or (len(held_ids) % cfg_loo.save_every == 0):
                _save_partial(partial_path, preds, trues, held_ids)

        finally:
            _cleanup_after_hold(
                model=model,
                state=state,
                out=out,
                train_mask=train_mask,
                run_dir=run_dir,
                keep_hold_dirs=cfg_loo.keep_hold_dirs,
                logger=logger,
            )

    _save_partial(partial_path, preds, trues, held_ids)

    return np.vstack(preds), np.vstack(trues), held_ids
