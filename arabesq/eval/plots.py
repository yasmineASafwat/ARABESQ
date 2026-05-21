
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, auc

try:
    from matplotlib_venn import venn3
except Exception:
    venn3 = None

from arabesq.eval.evaluate import CLASSES_4

def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_4)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=None, values_format="d", colorbar=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def save_venn_edges(n_sim: int, n_ppi: int, n_coex: int,
                    n_sim_ppi: int, n_sim_coex: int, n_ppi_coex: int, n_all: int,
                    out_path: str) -> None:
    if venn3 is None:
        return
    fig, ax = plt.subplots(figsize=(6,6))
    venn3(
        subsets=(n_sim - n_sim_ppi - n_sim_coex + n_all,
                 n_ppi - n_sim_ppi - n_ppi_coex + n_all,
                 n_sim_ppi - n_all,
                 n_coex - n_sim_coex - n_ppi_coex + n_all,
                 n_sim_coex - n_all,
                 n_ppi_coex - n_all,
                 n_all),
        set_labels=("SIM","PPI","COEX"),
        ax=ax
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
