
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Two supported backends:
# 1) facebookresearch/esm (fair-esm)
# 2) HuggingFace transformers
#
# We default to fair-esm if available.

@dataclass
class ESM2Config:
    model_name: str = "esm2_t33_650M_UR50D"  # changeable
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    nterm_len: int = 60
    layer: Optional[int] = None  # if None, use last layer
    batch_size: int = 4

def _load_esm_model(model_name: str):
    import esm  # type: ignore
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    return model, alphabet

@torch.no_grad()
def embed_sequences_fairesm(
    seqs: List[Tuple[str,str]],
    cfg: ESM2Config,
) -> Dict[str, np.ndarray]:
    """Return dict: gene -> concatenated [global_mean || nterm_mean]."""
    model, alphabet = _load_esm_model(cfg.model_name)
    model = model.to(cfg.device)
    batch_converter = alphabet.get_batch_converter()

    # choose layer
    if cfg.layer is None:
        repr_layer = model.num_layers
    else:
        repr_layer = cfg.layer

    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(seqs), cfg.batch_size):
        batch = seqs[i:i+cfg.batch_size]
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(cfg.device)

        res = model(toks, repr_layers=[repr_layer], return_contacts=False)
        reps = res["representations"][repr_layer]  # (B, T, C)

        # tokens: 0=<cls>, then sequence, then <eos>, then padding
        for b, (gene, s) in enumerate(batch):
            # length excludes special tokens
            L = len(s)
            # positions 1..L inclusive correspond to residues
            residue = reps[b, 1:1+L, :]  # (L, C)
            global_mean = residue.mean(dim=0)
            nL = min(cfg.nterm_len, L)
            nterm_mean = residue[:nL].mean(dim=0)
            vec = torch.cat([global_mean, nterm_mean], dim=0).detach().cpu().numpy().astype(np.float32)
            out[gene] = vec
    return out

def embed_sequences(
    seqs: List[Tuple[str,str]],
    cfg: ESM2Config,
    backend: str = "fairesm",
) -> Dict[str, np.ndarray]:
    if backend == "fairesm":
        return embed_sequences_fairesm(seqs, cfg)
    raise ValueError(f"Unsupported backend: {backend}")

def save_embeddings_npz(emb: Dict[str, np.ndarray], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **emb)

def load_embeddings_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
