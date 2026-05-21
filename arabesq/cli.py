from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch

from arabesq.utils.logging import setup_logger
from arabesq.utils.seed import set_seed
from arabesq.utils.ids import normalize_gene_id

from arabesq.io.ingestion import build_tables, split_train_val
from arabesq.io.suba import read_ppi, derive_ppi_confidence, read_coex, coex_weight
from arabesq.featurize.esm2_embed import ESM2Config, embed_sequences, save_embeddings_npz, load_embeddings_npz
from arabesq.graph.build import knn_edges, build_hetero_graph
from arabesq.graph.metrics import compute_graph_stats, to_dict
from arabesq.model.arabesq_gnn import arabesqModelConfig, arabesqGNN
from arabesq.train.trainer import TrainConfig, train_model
from arabesq.train.calibrate import fit_isotonic, save_calibrator, load_calibrator
from arabesq.eval.evaluate import (
    normalize_label_4class, y4_to_multihot3, y4_to_onehot4,
    map_probs_to_4class, overall_metrics, per_class_metrics, CLASSES_4
)
from arabesq.eval.plots import save_confusion, save_venn_edges
from arabesq.eval.loo import loo_strict, LOOConfig


def _collect_union(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_all = pd.concat([x for x in [train_df, val_df, test_df] if x is not None], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Gene"], keep="first")
    return df_all


def _make_masks(df_all: pd.DataFrame, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame]):
    ids = df_all["Gene"].tolist()
    idx = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for g in train_df["Gene"].tolist():
        if g in idx:
            train_mask[idx[g]] = True
    if val_df is not None:
        for g in val_df["Gene"].tolist():
            if g in idx:
                val_mask[idx[g]] = True
    if test_df is not None:
        for g in test_df["Gene"].tolist():
            if g in idx:
                test_mask[idx[g]] = True
    return train_mask, val_mask, test_mask


def _load_or_embed(df_all: pd.DataFrame, args, logger) -> np.ndarray:
    if args.embeddings_npz and Path(args.embeddings_npz).exists():
        logger.info(f"Loading embeddings: {args.embeddings_npz}")
        emb = load_embeddings_npz(args.embeddings_npz)
    else:
        seqs = list(zip(df_all["Gene"].tolist(), df_all["sequence"].tolist()))
        cfg = ESM2Config(
            model_name=args.esm2_model,
            device=args.device,
            nterm_len=args.nterm_len,
            batch_size=args.batch_size,
        )
        logger.info(f"Embedding {len(seqs)} sequences with {cfg.model_name} on {cfg.device}")
        emb = embed_sequences(seqs, cfg, backend="fairesm")

    X = np.stack([emb[g] for g in df_all["Gene"].tolist()], axis=0)
    return X


def _read_edges(args, logger):
    # Shared connect weighting (SUBA semantics). Defaults: match=1, adjacent=0.5, distant=0, unclear=0.
    connect_weights = None
    if getattr(args, "connect_weights", None):
        connect_weights = {}
        for part in str(args.connect_weights).split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            connect_weights[k.strip().upper()] = float(v)

    ppi_pairs = None
    if args.ppi_csv:
        df_ppi = read_ppi(
            args.ppi_csv,
            locusA_col=args.ppi_a_col,
            locusB_col=args.ppi_b_col,
            connect_col=args.ppi_connect_col,
        )
        ppi_pairs = derive_ppi_confidence(
            df_ppi,
            method=args.ppi_conf_method,
            conf_col=args.ppi_conf_col,
            connect_weights=connect_weights,
        )
        logger.info(f"Loaded PPI undirected pairs: {len(ppi_pairs)}")

    coex_pairs = None
    if args.coex_csv:
        df_coex = read_coex(
            args.coex_csv,
            locusA_col=args.coex_a_col,
            locusB_col=args.coex_b_col,
            rank_col=args.coex_rank_col,
            ave_col=args.coex_ave_col,
            connect_col=args.coex_connect_col,
        )
        coex_pairs = coex_weight(
            df_coex,
            rank_mode=args.coex_rank_mode,
            weight_col=args.coex_weight_col,
            connect_weights=connect_weights,
            combine=args.coex_combine,
        )
        logger.info(f"Loaded COEX undirected pairs: {len(coex_pairs)}")

    return ppi_pairs, coex_pairs


def _ensure_val_split(tables, args, logger):
    """If val_csv wasn't provided, create val from train using --val_frac."""
    if tables.val is not None and len(tables.val) > 0:
        return tables
    logger.info("val_csv not provided; creating validation split from train.")
    tr, va = split_train_val(
        tables.train,
        val_frac=args.val_frac,
        mode=args.val_split_mode,
        seed=args.seed,
        group_col=args.group_col,
        stratify_col="localization",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out_dir / "train_split.csv", index=False)
    va.to_csv(out_dir / "val_split.csv", index=False)
    return type(tables)(train=tr, val=va, test=tables.test)


def _build_graph(df_all: pd.DataFrame, X: np.ndarray, args, ppi_pairs, coex_pairs):
    edge_sim = knn_edges(X, k=args.knn_k, self_loops=False)
    proteins = df_all["Gene"].tolist()
    data = build_hetero_graph(
        proteins=proteins,
        X=X,
        edges_sim=edge_sim,
        ppi_pairs=ppi_pairs if args.use_ppi else None,
        coex_pairs=coex_pairs if args.use_coex else None,
        coex_gate_baseline=args.coex_gate,
        gate_tau_sim=args.coex_gate_tau_sim,
    )
    return data


def cmd_embed(args):
    logger = setup_logger(args.out_dir)
    set_seed(args.seed)

    tables = build_tables(args.train_csv, args.val_csv, args.test_csv)
    df_all = _collect_union(tables.train, tables.val, tables.test)
    seqs = list(zip(df_all["Gene"].tolist(), df_all["sequence"].tolist()))

    cfg = ESM2Config(model_name=args.esm2_model, device=args.device, nterm_len=args.nterm_len, batch_size=args.batch_size)
    logger.info(f"Embedding {len(seqs)} sequences with {cfg.model_name} on {cfg.device}")
    emb = embed_sequences(seqs, cfg, backend="fairesm")
    out_path = str(Path(args.out_dir) / "esm2_embeddings.npz")
    save_embeddings_npz(emb, out_path)
    logger.info(f"Saved embeddings to {out_path}")


def _prepare_targets(df_all: pd.DataFrame, out_mode: str):
    """Prepare training targets.

    out_mode:
      - multilabel: y float (N,3) multi-hot [mito, plastid, other]
      - softmax4:   y long  (N,)  class index in {0,1,2,3}
      - dirichlet4: y float (N,4) one-hot

    Returns:
      y: tensor (see above)
      y4: int (N,) canonical 4-class indices
      y_cal: float (N,3) targets for isotonic calibration (multilabel only)
    """
    y4 = np.array([normalize_label_4class(x) for x in df_all["localization"].tolist()], dtype=int)
    y_multihot3 = y4_to_multihot3(y4)
    y_onehot4 = y4_to_onehot4(y4)

    if out_mode == "multilabel":
        y = torch.tensor(y_multihot3, dtype=torch.float32)
        y_cal = y_multihot3
    elif out_mode == "softmax4":
        y = torch.tensor(y4, dtype=torch.long)
        y_cal = None
    elif out_mode == "dirichlet4":
        y = torch.tensor(y_onehot4, dtype=torch.float32)
        y_cal = None
    else:
        raise ValueError(out_mode)

    return y, y4, y_cal


def _save_test_predictions(df_all: pd.DataFrame, mask: np.ndarray, y_true4: np.ndarray, y_pred4: np.ndarray,
                           probs: np.ndarray, out_mode: str, out_path: Path,
                           extra: Optional[Dict[str, np.ndarray]] = None):
    """Save per-protein predictions.

    - multilabel: `probs` is (N,3) marginals (p_mito, p_plastid, p_other)
    - softmax4/dirichlet4: `probs` is (N,4) (p_mito_only, p_plastid_only, p_dual, p_other)

    `extra` can be used to attach additional per-protein numeric columns (e.g., uncertainty).
    """
    genes = df_all["Gene"].to_numpy()[mask]

    base = {
        "Gene": genes,
        "y_true": [CLASSES_4[i] for i in y_true4],
        "y_pred": [CLASSES_4[i] for i in y_pred4],
        "correct": (y_true4 == y_pred4).astype(int),
    }

    if out_mode == "multilabel":
        base.update({
            "p_mito": probs[:, 0],
            "p_plastid": probs[:, 1],
            "p_other": probs[:, 2],
        })
    else:
        base.update({
            "p_mito_only": probs[:, 0],
            "p_plastid_only": probs[:, 1],
            "p_dual": probs[:, 2],
            "p_other": probs[:, 3],
            "p_mito": probs[:, 0] + probs[:, 2],
            "p_plastid": probs[:, 1] + probs[:, 2],
        })

    if extra:
        for k, v in extra.items():
            base[k] = v

    pd.DataFrame(base).to_csv(out_path, index=False)


def _edge_overlap_counts(data) -> Optional[dict]:
    # Only meaningful when all three relations exist
    if ("protein", "ppi", "protein") not in data.edge_types and ("protein", "coex", "protein") not in data.edge_types:
        return None

    def edgeset(et):
        if et not in data.edge_types:
            return set()
        ei = data[et].edge_index.cpu().numpy()
        return set(zip(ei[0].tolist(), ei[1].tolist()))

    sim = edgeset(("protein", "sim", "protein"))
    ppi = edgeset(("protein", "ppi", "protein"))
    coe = edgeset(("protein", "coex", "protein"))
    return {
        "n_sim": len(sim),
        "n_ppi": len(ppi),
        "n_coex": len(coe),
        "n_sim_ppi": len(sim & ppi),
        "n_sim_coex": len(sim & coe),
        "n_ppi_coex": len(ppi & coe),
        "n_all": len(sim & ppi & coe),
    }


def cmd_train(args):
    logger = setup_logger(args.out_dir)
    set_seed(args.seed)

    tables = build_tables(args.train_csv, args.val_csv, args.test_csv)
    tables = _ensure_val_split(tables, args, logger)
    df_all = _collect_union(tables.train, tables.val, tables.test)
    train_mask, val_mask, test_mask = _make_masks(df_all, tables.train, tables.val, tables.test)
    if val_mask.sum() == 0:
        raise ValueError("Validation split is empty; check --val_frac or your input CSV.")

    X = _load_or_embed(df_all, args, logger)
    ppi_pairs, coex_pairs = _read_edges(args, logger)
    data = _build_graph(df_all, X, args, ppi_pairs, coex_pairs)

    stats = compute_graph_stats(data)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "graph_stats.json").write_text(json.dumps(to_dict(stats), indent=2))
    logger.info(f"Graph stats: {stats}")

    overlap = _edge_overlap_counts(data)
    if overlap:
        (Path(args.out_dir) / "edge_overlap.json").write_text(json.dumps(overlap, indent=2))
        save_venn_edges(**overlap, out_path=str(Path(args.out_dir) / "edge_overlap_venn.png"))

    y, y4, y_cal = _prepare_targets(df_all, args.out_mode)

    mcfg = arabesqModelConfig(
        in_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        relation_dropout=args.relation_dropout,
        out_mode=args.out_mode,
        ppi_high_th=args.ppi_high_th,
        use_evidence_tiers=(not args.no_evidence_tiers),
        bilinear_gate_dim=args.bilinear_gate_dim,
    )
    model = arabesqGNN(mcfg)

    tcfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        out_mode=args.out_mode,
    )

    ckpt = train_model(model, data, y, train_mask, val_mask, tcfg, args.out_dir, logger)
    logger.info(f"Best checkpoint: {ckpt}")

    # Inference on full graph
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model_state"])
    model = model.to(args.device).eval()
    with torch.no_grad():
        out = model(data.to(args.device))

    beta12 = out["semantic_weights_tier12_mean"].detach().cpu().numpy().tolist()
    beta3 = out["semantic_weights_tier3_mean"].detach().cpu().numpy().tolist()
    (Path(args.out_dir) / "semantic_attention.json").write_text(
        json.dumps({"tier12_relations": ["sim", "ppi_high"], "tier12_mean": beta12,
                    "tier3_relations": ["ppi_low", "coex"], "tier3_mean": beta3}, indent=2)
    )

    if test_mask.sum() == 0:
        return

    test_np = test_mask.cpu().numpy()
    y_true = y4[test_np]

    if args.out_mode == "multilabel":
        probs_marginal3 = out["probs_marginal3"].detach().cpu().numpy()
        cal = fit_isotonic(probs_marginal3[val_mask.cpu().numpy()], y_cal[val_mask.cpu().numpy()])
        cal_path = str(Path(args.out_dir) / "isotonic.joblib")
        save_calibrator(cal, cal_path)
        logger.info(f"Saved calibrator: {cal_path}")

        probs_cal = cal.predict(probs_marginal3)
        y_pred = map_probs_to_4class(
            probs_cal[test_np],
            th_mito=args.th_mito,
            th_plastid=args.th_plastid,
            th_other=args.th_other,
        )

        overall = overall_metrics(y_true, y_pred)
        percls = per_class_metrics(y_true, y_pred)

        Path(args.out_dir, "metrics_overall.json").write_text(json.dumps(overall, indent=2))
        percls.to_csv(Path(args.out_dir, "metrics_per_class.csv"), index=False)
        save_confusion(y_true, y_pred, str(Path(args.out_dir) / "confusion.png"))

        _save_test_predictions(df_all, test_np, y_true, y_pred, probs_cal[test_np], "multilabel", Path(args.out_dir) / "test_predictions.csv")

    else:
        probs4 = out["probs4"].detach().cpu().numpy()
        y_pred = probs4[test_np].argmax(axis=1).astype(int)
        extra = None
        if args.out_mode == "dirichlet4" and "uncertainty" in out:
            extra = {"uncertainty": out["uncertainty"].detach().cpu().numpy()[test_np]}

        overall = overall_metrics(y_true, y_pred)
        percls = per_class_metrics(y_true, y_pred)

        Path(args.out_dir, "metrics_overall.json").write_text(json.dumps(overall, indent=2))
        percls.to_csv(Path(args.out_dir, "metrics_per_class.csv"), index=False)
        save_confusion(y_true, y_pred, str(Path(args.out_dir) / "confusion.png"))

        _save_test_predictions(df_all, test_np, y_true, y_pred, probs4[test_np], args.out_mode, Path(args.out_dir) / "test_predictions.csv", extra=extra)
        (Path(args.out_dir) / "calibration_skipped.txt").write_text(
            "Calibration skipped for out_mode != multilabel. Consider temperature scaling for softmax4/dirichlet4.\n"
        )

    (Path(args.out_dir) / "misclassified.csv").write_text(
        "\n".join(df_all.loc[test_np, "Gene"][y_true != y_pred].tolist())
    )
    logger.info(f"Test overall: {overall}")


def cmd_benchmark(args):
    logger = setup_logger(args.out_dir)
    set_seed(args.seed)

    tables = build_tables(args.train_csv, args.val_csv, args.test_csv)
    tables = _ensure_val_split(tables, args, logger)
    df_all = _collect_union(tables.train, tables.val, tables.test)
    train_mask, val_mask, test_mask = _make_masks(df_all, tables.train, tables.val, tables.test)
    if val_mask.sum() == 0:
        raise ValueError("benchmark requires a non-empty validation split.")
    if test_mask.sum() == 0:
        raise ValueError("benchmark requires test_csv.")

    X = _load_or_embed(df_all, args, logger)
    ppi_pairs, coex_pairs = _read_edges(args, logger)

    setups = [
        ("SIM", False, False),
        ("SIM_PPI", True, False),
        ("SIM_COEX", False, True),
        ("SIM_PPI_COEX", True, True),
    ]

    y, y4, y_cal = _prepare_targets(df_all, args.out_mode)
    ablation_rows = []

    for name, use_ppi, use_coex in setups:
        out_dir = Path(args.out_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== Setup: {name} ===")

        args.use_ppi = use_ppi
        args.use_coex = use_coex

        data = _build_graph(df_all, X, args, ppi_pairs, coex_pairs)
        stats = compute_graph_stats(data)
        (out_dir / "graph_stats.json").write_text(json.dumps(to_dict(stats), indent=2))

        overlap = _edge_overlap_counts(data)
        if overlap:
            (out_dir / "edge_overlap.json").write_text(json.dumps(overlap, indent=2))
            save_venn_edges(**overlap, out_path=str(out_dir / "edge_overlap_venn.png"))

        mcfg = arabesqModelConfig(
            in_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            relation_dropout=args.relation_dropout,
            out_mode=args.out_mode,
            ppi_high_th=args.ppi_high_th,
            use_evidence_tiers=(not args.no_evidence_tiers),
            bilinear_gate_dim=args.bilinear_gate_dim,
        )
        model = arabesqGNN(mcfg)

        tcfg = TrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
            out_mode=args.out_mode,
        )

        ckpt = train_model(model, data, y, train_mask, val_mask, tcfg, str(out_dir), logger)

        # Inference
        state = torch.load(ckpt, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        model = model.to(args.device).eval()
        with torch.no_grad():
            out = model(data.to(args.device))

        beta12 = out["semantic_weights_tier12_mean"].detach().cpu().numpy().tolist()
        beta3 = out["semantic_weights_tier3_mean"].detach().cpu().numpy().tolist()
        (out_dir / "semantic_attention.json").write_text(
            json.dumps({"tier12_relations": ["sim", "ppi_high"], "tier12_mean": beta12,
                        "tier3_relations": ["ppi_low", "coex"], "tier3_mean": beta3}, indent=2)
        )

        test_np = test_mask.cpu().numpy()
        y_true = y4[test_np]

        if args.out_mode == "multilabel":
            probs_marginal3 = out["probs_marginal3"].detach().cpu().numpy()
            cal = fit_isotonic(probs_marginal3[val_mask.cpu().numpy()], y_cal[val_mask.cpu().numpy()])
            save_calibrator(cal, str(out_dir / "isotonic.joblib"))
            probs_cal = cal.predict(probs_marginal3)

            y_pred = map_probs_to_4class(
                probs_cal[test_np],
                th_mito=args.th_mito,
                th_plastid=args.th_plastid,
                th_other=args.th_other,
            )

            overall = overall_metrics(y_true, y_pred)
            percls = per_class_metrics(y_true, y_pred)

            (out_dir / "metrics_overall.json").write_text(json.dumps(overall, indent=2))
            percls.to_csv(out_dir / "metrics_per_class.csv", index=False)
            save_confusion(y_true, y_pred, str(out_dir / "confusion.png"))

            _save_test_predictions(df_all, test_np, y_true, y_pred, probs_cal[test_np], "multilabel", out_dir / "test_predictions.csv")

        else:
            probs4 = out["probs4"].detach().cpu().numpy()
            y_pred = probs4[test_np].argmax(axis=1).astype(int)
            extra = None
            if args.out_mode == "dirichlet4" and "uncertainty" in out:
                extra = {"uncertainty": out["uncertainty"].detach().cpu().numpy()[test_np]}

            overall = overall_metrics(y_true, y_pred)
            percls = per_class_metrics(y_true, y_pred)

            (out_dir / "metrics_overall.json").write_text(json.dumps(overall, indent=2))
            percls.to_csv(out_dir / "metrics_per_class.csv", index=False)
            save_confusion(y_true, y_pred, str(out_dir / "confusion.png"))

            _save_test_predictions(df_all, test_np, y_true, y_pred, probs4[test_np], args.out_mode, out_dir / "test_predictions.csv", extra=extra)
            (out_dir / "calibration_skipped.txt").write_text(
            "Calibration skipped for out_mode != multilabel. Consider temperature scaling for softmax4/dirichlet4.\n"
            )

        row = {"setup": name, **overall}
        ablation_rows.append(row)
        logger.info(f"{name} Test overall: {overall}")

    pd.DataFrame(ablation_rows).to_csv(Path(args.out_dir) / "ablation_summary.csv", index=False)


def cmd_predict(args):
    logger = setup_logger(Path(args.out_csv).parent)

    df = pd.read_csv(args.input_csv)
    if "Gene" not in df.columns or "sequence" not in df.columns:
        raise ValueError("input_csv must contain at least Gene and sequence columns.")

    df = df.copy()
    df["Gene"] = df["Gene"].map(normalize_gene_id)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    cfg = ESM2Config(model_name=args.esm2_model, device=args.device, nterm_len=args.nterm_len, batch_size=args.batch_size)
    emb = embed_sequences(list(zip(df["Gene"].tolist(), df["sequence"].tolist())), cfg)
    X = np.stack([emb[g] for g in df["Gene"].tolist()], axis=0)

    edge_sim = knn_edges(X, k=args.knn_k, self_loops=False)
    data = build_hetero_graph(df["Gene"].tolist(), X, edge_sim, None, None, coex_gate_baseline=False)

    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    mcfg_obj = state.get("cfg", None)

    def cfg_get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if mcfg_obj is None:
        mcfg = arabesqModelConfig(
            in_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            relation_dropout=0.0,
            out_mode=args.out_mode,
            ppi_high_th=args.ppi_high_th,
            use_evidence_tiers=(not args.no_evidence_tiers),
            bilinear_gate_dim=args.bilinear_gate_dim,
        )
    else:
        mcfg = arabesqModelConfig(
            in_dim=X.shape[1],
            hidden_dim=int(cfg_get(mcfg_obj, "hidden_dim", args.hidden_dim)),
            heads=int(cfg_get(mcfg_obj, "heads", args.heads)),
            dropout=float(cfg_get(mcfg_obj, "dropout", args.dropout)),
            relation_dropout=0.0,
            out_mode=str(cfg_get(mcfg_obj, "out_mode", args.out_mode)),
            ppi_high_th=float(cfg_get(mcfg_obj, "ppi_high_th", args.ppi_high_th)),
            use_evidence_tiers=bool(cfg_get(mcfg_obj, "use_evidence_tiers", (not args.no_evidence_tiers))),
            bilinear_gate_dim=int(cfg_get(mcfg_obj, "bilinear_gate_dim", args.bilinear_gate_dim)),
        )

    model = arabesqGNN(mcfg)
    model.load_state_dict(state["model_state"])
    model = model.to(args.device).eval()

    with torch.no_grad():
        out = model(data.to(args.device))

    df_out = df.copy()

    if mcfg.out_mode == "multilabel":
        probs = out["probs_marginal3"].detach().cpu().numpy()
        if args.calibrator and Path(args.calibrator).exists():
            cal = load_calibrator(args.calibrator)
            probs = cal.predict(probs)

        pred4 = map_probs_to_4class(probs, th_mito=args.th_mito, th_plastid=args.th_plastid, th_other=args.th_other)
        df_out["p_mito"] = probs[:, 0]
        df_out["p_plastid"] = probs[:, 1]
        df_out["p_other"] = probs[:, 2]
        df_out["pred_class"] = ["mito" if i == 0 else "plastid" if i == 1 else "dual" if i == 2 else "other" for i in pred4]

    else:
        probs4 = out["probs4"].detach().cpu().numpy()
        if args.calibrator:
            logger.info("Ignoring --calibrator for out_mode != multilabel (use temperature scaling instead).")
        pred4 = probs4.argmax(axis=1).astype(int)
        df_out["p_mito_only"] = probs4[:, 0]
        df_out["p_plastid_only"] = probs4[:, 1]
        df_out["p_dual"] = probs4[:, 2]
        df_out["p_other"] = probs4[:, 3]
        df_out["p_mito"] = probs4[:, 0] + probs4[:, 2]
        df_out["p_plastid"] = probs4[:, 1] + probs4[:, 2]
        df_out["pred_class"] = ["mito" if i == 0 else "plastid" if i == 1 else "dual" if i == 2 else "other" for i in pred4]

    df_out.to_csv(args.out_csv, index=False)
    logger.info(f"Saved predictions to {args.out_csv}")


def cmd_loo(args):
    logger = setup_logger(args.out_dir)
    set_seed(args.seed)

    tables = build_tables(args.train_csv, args.val_csv, args.test_csv)
    tables = _ensure_val_split(tables, args, logger)
    df_all = _collect_union(tables.train, tables.val, tables.test)
    train_mask, val_mask, _ = _make_masks(df_all, tables.train, tables.val, tables.test)
    if val_mask.sum() == 0:
        raise ValueError("loo requires a non-empty validation split.")

    X = _load_or_embed(df_all, args, logger)
    ppi_pairs, coex_pairs = _read_edges(args, logger)
    data = _build_graph(df_all, X, args, ppi_pairs, coex_pairs)

    y, y4, _ = _prepare_targets(df_all, args.out_mode)

    def ctor():
        mcfg = arabesqModelConfig(
            in_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            relation_dropout=args.relation_dropout,
            out_mode=args.out_mode,
            ppi_high_th=args.ppi_high_th,
            use_evidence_tiers=(not args.no_evidence_tiers),
            bilinear_gate_dim=args.bilinear_gate_dim,
        )
        return arabesqGNN(mcfg)

    tcfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        out_mode=args.out_mode,
    )

    preds, trues, held_ids = loo_strict(
        model_ctor=ctor,
        base_data=data,
        y=y,
        base_train_mask=train_mask,
        val_mask=val_mask,
        cfg_train=tcfg,
        cfg_loo=LOOConfig(strict=True, max_holds=args.max_holds),
        out_dir=args.out_dir,
        logger=logger,
    )

    y_true4 = y4[np.array(held_ids)]
    if args.out_mode == "multilabel":
        y_pred4 = map_probs_to_4class(preds, th_mito=args.th_mito, th_plastid=args.th_plastid, th_other=args.th_other)
    else:
        y_pred4 = preds.argmax(axis=1).astype(int)

    overall = overall_metrics(y_true4, y_pred4)
    percls = per_class_metrics(y_true4, y_pred4)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "loo_metrics_overall.json").write_text(json.dumps(overall, indent=2))
    percls.to_csv(Path(args.out_dir) / "loo_metrics_per_class.csv", index=False)
    save_confusion(y_true4, y_pred4, str(Path(args.out_dir) / "loo_confusion.png"))
    logger.info(f"LOO overall: {overall}")


def build_parser():
    p = argparse.ArgumentParser(prog="arabesq", description="arabesq: multi-relational GNN for dual-targeted protein prediction")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(a):
        a.add_argument("--seed", type=int, default=1337)
        a.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        a.add_argument("--esm2_model", type=str, default="esm2_t33_650M_UR50D")
        a.add_argument("--nterm_len", type=int, default=60)
        a.add_argument("--batch_size", type=int, default=4)
        a.add_argument("--embeddings_npz", type=str, default=None, help="Optional precomputed embeddings npz.")

        a.add_argument("--knn_k", type=int, default=20)

        # Validation split (used when --val_csv is not provided)
        a.add_argument("--val_frac", type=float, default=0.1, help="If val_csv is omitted, hold out this fraction from train.")
        a.add_argument(
            "--val_split_mode",
            type=str,
            default="auto",
            choices=["auto", "stratified", "cluster", "cluster_stratified"],
            help="How to form validation split from train when val_csv is omitted.",
        )
        a.add_argument("--group_col", type=str, default="cluster", help="Group column for cluster-aware splitting.")

        # Generic PPI/COEX support (SUBA, STRING, etc.)
        a.add_argument("--ppi_csv", type=str, default=None)
        a.add_argument("--ppi_a_col", type=str, default="locusA")
        a.add_argument("--ppi_b_col", type=str, default="locusB")
        a.add_argument("--ppi_connect_col", type=str, default="connect", help="Optional spatial category column (SUBA).")
        a.add_argument("--ppi_conf_col", type=str, default=None, help="If provided, use this numeric column as confidence (0..1).")
        a.add_argument("--ppi_conf_method", type=str, default="paper_count", choices=["paper_count", "binary"])

        a.add_argument("--coex_csv", type=str, default=None)
        a.add_argument("--coex_a_col", type=str, default="locusA")
        a.add_argument("--coex_b_col", type=str, default="locusB")
        a.add_argument("--coex_rank_col", type=str, default="new_rank", help="Mutual-rank column (SUBA).")
        a.add_argument("--coex_ave_col", type=str, default="ave", help="Scaled correlation-like column (SUBA).")
        a.add_argument("--coex_connect_col", type=str, default="connect", help="Spatial category column (SUBA).")
        a.add_argument("--coex_weight_col", type=str, default=None, help="If provided, use this numeric column as COEX weight (0..1).")
        a.add_argument("--coex_rank_mode", type=str, default="lower_is_stronger", choices=["lower_is_stronger", "higher_is_stronger", "none"])
        a.add_argument("--coex_combine", type=str, default="geom", choices=["geom", "mean", "product"], help="How to combine MR and ave into a base COEX weight.")
        a.add_argument("--coex_gate", action="store_true")
        a.add_argument("--coex_gate_tau_sim", type=float, default=0.25)

        a.add_argument(
            "--connect_weights",
            type=str,
            default=None,
            help="Override spatial weights, e.g. 'match=1,adjacent=0.5,distant=0,unclear=0'.",
        )

        # Evidence-tier controls
        a.add_argument("--ppi_high_th", type=float, default=0.7, help="PPI confidence threshold for Tier-2 (high-confidence) edges.")
        a.add_argument("--no_evidence_tiers", action="store_true", help="Disable Tier-3 contextual evidence processing (COEX + low-conf PPI).")
        a.add_argument("--bilinear_gate_dim", type=int, default=64, help="Projection dim for COEX bilinear integration gate.")

        a.add_argument("--hidden_dim", type=int, default=256)
        a.add_argument("--heads", type=int, default=4)
        a.add_argument("--dropout", type=float, default=0.2)
        a.add_argument("--relation_dropout", type=float, default=0.2)
        a.add_argument("--out_mode", type=str, default="multilabel", choices=["multilabel", "softmax4", "dirichlet4"])

        a.add_argument("--lr", type=float, default=1e-3)
        a.add_argument("--weight_decay", type=float, default=1e-4)
        a.add_argument("--max_epochs", type=int, default=100)
        a.add_argument("--patience", type=int, default=15)

        # Thresholds used only in multilabel decision mapping
        a.add_argument("--th_mito", type=float, default=0.5)
        a.add_argument("--th_plastid", type=float, default=0.5)
        a.add_argument("--th_other", type=float, default=0.5)

    pe = sub.add_parser("embed", help="Precompute ESM-2 embeddings to npz")
    pe.add_argument("--train_csv", required=True)
    pe.add_argument("--val_csv", default=None)
    pe.add_argument("--test_csv", default=None)
    pe.add_argument("--out_dir", required=True)
    add_common(pe)
    pe.set_defaults(func=cmd_embed)

    pt = sub.add_parser("train", help="Train a single setup (choose --use_ppi / --use_coex)")
    pt.add_argument("--train_csv", required=True)
    pt.add_argument("--val_csv", default=None)
    pt.add_argument("--test_csv", default=None)
    pt.add_argument("--out_dir", required=True)
    pt.add_argument("--use_ppi", action="store_true")
    pt.add_argument("--use_coex", action="store_true")
    add_common(pt)
    pt.set_defaults(func=cmd_train)

    pb = sub.add_parser("benchmark", help="Run ablations: SIM, SIM+PPI, SIM+COEX, SIM+PPI+COEX")
    pb.add_argument("--train_csv", required=True)
    pb.add_argument("--val_csv", default=None)
    pb.add_argument("--test_csv", required=True)
    pb.add_argument("--out_dir", required=True)
    add_common(pb)
    pb.set_defaults(func=cmd_benchmark)

    pp = sub.add_parser("predict", help="Predict on new proteins (similarity-only graph by default)")
    pp.add_argument("--ckpt", required=True)
    pp.add_argument("--input_csv", required=True)
    pp.add_argument("--out_csv", required=True)
    pp.add_argument("--calibrator", default=None)
    add_common(pp)
    pp.set_defaults(func=cmd_predict)

    pl = sub.add_parser("loo", help="Strict Leave-One-Out CV (retrain per held-out protein; expensive)")
    pl.add_argument("--train_csv", required=True)
    pl.add_argument("--val_csv", default=None)
    pl.add_argument("--test_csv", default=None)
    pl.add_argument("--out_dir", required=True)
    pl.add_argument("--use_ppi", action="store_true")
    pl.add_argument("--use_coex", action="store_true")
    pl.add_argument("--max_holds", type=int, default=None, help="Limit # held-outs for a smoke-test.")
    add_common(pl)
    pl.set_defaults(func=cmd_loo)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
