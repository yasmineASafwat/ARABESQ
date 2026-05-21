
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

CLASSES_4 = ["mito", "plastid", "dual", "other"]

def map_probs_to_4class(
    probs_marginal3: np.ndarray,
    th_mito: float = 0.5,
    th_plastid: float = 0.5,
    th_other: float = 0.5,
) -> np.ndarray:
    """Threshold-based decision logic:

    If Pmito>th_mito and Pplastid>th_plastid => dual
    elif Pmito>th_mito => mito
    elif Pplastid>th_plastid => plastid
    elif Pother>th_other => other
    else => other (conservative default)
    """
    p0, p1, p2 = probs_marginal3[:, 0], probs_marginal3[:, 1], probs_marginal3[:, 2]
    pred = np.full(len(probs_marginal3), 3, dtype=int)  # other
    pred[(p0 > th_mito) & (p1 > th_plastid)] = 2
    pred[(p0 > th_mito) & ~(p1 > th_plastid)] = 0
    pred[(p1 > th_plastid) & ~(p0 > th_mito)] = 1
    pred[(p2 > th_other) & (pred == 3)] = 3
    return pred

def normalize_label_4class(loc: str) -> int:
    s = str(loc).strip().lower()
    if "dual" in s or ("mito" in s and ("plast" in s or "chloro" in s)):
        return 2
    if "mito" in s:
        return 0
    if "plast" in s or "chloro" in s:
        return 1
    return 3

def y4_to_multihot3(y4: np.ndarray) -> np.ndarray:
    """Convert 4-class labels into 3 binary labels (mito, plastid, other). Dual is (1,1,0)."""
    y = np.zeros((len(y4), 3), dtype=np.float32)
    y[y4 == 0, 0] = 1
    y[y4 == 1, 1] = 1
    y[y4 == 2, 0] = 1
    y[y4 == 2, 1] = 1
    y[y4 == 3, 2] = 1
    return y

def y4_to_onehot4(y4: np.ndarray) -> np.ndarray:
    y = np.zeros((len(y4), 4), dtype=np.float32)
    y[np.arange(len(y4)), y4] = 1.0
    return y

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    out = pd.DataFrame({
        "class": CLASSES_4,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": sup,
    })
    out["accuracy"] = [
        float((((y_true == c) & (y_pred == c)).sum()) / max(1, (y_true == c).sum()))
        for c in [0, 1, 2, 3]
    ]
    return out

def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_m),
        "macro_recall": float(rec_m),
        "macro_f1": float(f1_m),
        "weighted_precision": float(prec_w),
        "weighted_recall": float(rec_w),
        "weighted_f1": float(f1_w),
    }
