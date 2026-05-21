# ARABESQ (Dual-targeted protein localization with multi-relational GNN)

This repository is a production-ready **template** for the ARABESQ architecture described in the accompanying paper notebook.

Defaults are aligned to SUBA semantics for Arabidopsis:
- COEX `new_rank` = Mutual Rank (MR): **lower is stronger** (1 best, 100 weakest in subset)
- COEX `ave` ~ 1..25: **higher is stronger**
- `connect` ∈ {match, adjacent, distant, unclear} is used as a spatial prior:
  match=1.0, adjacent=0.5, distant=0.0, unclear=0.0

## Quickstart

```bash
# 1- create env (example)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2- train + evaluate all graph ablations (SIM, SIM+PPI, SIM+COEX, SIM+PPI+COEX)
python -m arabesq.cli benchmark \
  --train_csv path/to/train.csv \
  --test_csv  path/to/test.csv \
  --ppi_csv   path/to/suba_ppi.csv \
  --coex_csv  path/to/suba_coex.csv \
  --out_dir   runs/arabidopsis_run

# 3- predict on new proteins with a trained checkpoint
python -m arabesq.cli predict \
  --ckpt runs/arabidopsis_run/SIM_PPI_COEX/model_best.pt \
  --input_csv path/to/new_proteins.csv \
  --out_csv   preds.csv
```

## Inputs

### Train/Test CSV
Expected columns (minimum):
- `Gene` (locus / gene id)
- `localization` (mito|plastid|dual|other; flexible mapping)
- `sequence` (AA sequence)

Additional columns are preserved and passed through to outputs.

### SUBA COEX
Expected columns:
- `locusA, locusB, new_rank, ave, connect` (+ optional additional fields)

### SUBA PPI
Expected columns:
- `locusA, locusB, paper` (+ optional `connect`)

### STRING PPI (example)
If you use STRING, pass column names explicitly:

```bash
python -m arabesq.cli train \
  --train_csv train.csv --test_csv test.csv --out_dir runs/str \
  --ppi_csv string_ppi.csv --ppi_a_col source --ppi_b_col target --ppi_conf_col confidence \
  --use_ppi
```

### Validation
If you don't have a separate validation CSV, ARABESQ will create it from train:

```bash
python -m arabesq.cli train \
  --train_csv train.csv --test_csv test.csv --out_dir runs/x \
  --val_frac 0.1 --val_split_mode cluster_stratified --group_col cluster
```

## Notes on design choices
- Multi-relational GAT encoders + semantic-level attention fusion (HAN-style). 
- Edge attributes (PPI confidence, COEX weight) are injected into attention by **log-weight scaling**:
  e'ij = eij + log(sij + eps), so low-confidence edges are mathematically ignored.
- Optional evidential (Dirichlet) head for uncertainty; isotonic regression calibration on validation set.

See the notebook in `arabesq/notebooks/ARABESQ_Paper_Repro.ipynb`.
