# ARABESQ

**ARAbidopsis multi-graph Biological Evidence fuSion for dual-target protein Quantification**

ARABESQ is a heterogeneous multi-relational graph neural network (GNN) framework for predicting subcellular localization of Arabidopsis proteins, with a particular focus on identifying proteins dual-targeted to both mitochondria and plastids.

The framework integrates multiple complementary evidence sources:

- Sequence similarity derived from ESM-2 protein language model embeddings
- Protein–protein interaction (PPI) networks
- Co-expression (COEX) relationships
- Spatial compatibility priors derived from SUBA annotations

ARABESQ supports:

- Four-class prediction:
  - mitochondrion-only
  - plastid-only
  - dual-targeted
  - other
- Multilabel prediction
- Evidential uncertainty estimation
- Graph ablation benchmarking
- Probability calibration

---

# Repository Structure

```text
arabesq/
├── arabesq/                 # core package
├── notebooks/               # reproducibility notebooks
├── runs/                    # generated outputs
├── requirements.txt
├── README.md
└── setup.py
```

---

# Installation

## Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU acceleration is strongly recommended for ESM-2 embedding extraction.

---

# Quick Start

## Benchmark graph configurations

This command trains and evaluates all graph ablations:

- SIM
- SIM+PPI
- SIM+COEX
- SIM+PPI+COEX

```bash
python -m arabesq.cli benchmark \
  --train_csv path/to/train.csv \
  --test_csv path/to/test.csv \
  --ppi_csv path/to/suba_ppi.csv \
  --coex_csv path/to/suba_coex.csv \
  --out_dir runs/arabidopsis_run
```

---

## Predict on new proteins

```bash
python -m arabesq.cli predict \
  --ckpt runs/arabidopsis_run/SIM_PPI_COEX/model_best.pt \
  --input_csv path/to/new_proteins.csv \
  --out_csv preds.csv
```

---

# Input Data

## Train/Test CSV

Minimum required columns:

| Column | Description |
|---|---|
| `Gene` | Gene or protein identifier |
| `localization` | Ground-truth label (`mito`, `plastid`, `dual`, `other`) |
| `sequence` | Amino acid sequence |

Additional metadata columns are preserved and propagated to output prediction tables.

---

## SUBA Co-expression (COEX)

Expected columns:

```text
locusA, locusB, new_rank, ave, connect
```

### Default SUBA semantics

- `new_rank`
  - Mutual Rank (MR)
  - lower values indicate stronger co-expression

- `ave`
  - approximately 1–25
  - higher values indicate stronger co-expression

- `connect`
  - spatial compatibility prior:

| connect | weight |
|---|---|
| match | 1.0 |
| adjacent | 0.5 |
| distant | 0.0 |
| unclear | 0.0 |

Final COEX edge weights are computed using the geometric mean of normalized MR and `ave` scores and multiplied by the spatial compatibility prior.

---

## SUBA PPI

Expected columns:

```text
locusA, locusB, paper
```

Optional:

```text
connect
```

---

## STRING PPI Example

When using STRING-derived interaction networks, specify the appropriate column names explicitly:

```bash
python -m arabesq.cli train \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir runs/string_run \
  --ppi_csv string_ppi.csv \
  --ppi_a_col source \
  --ppi_b_col target \
  --ppi_conf_col confidence \
  --use_ppi
```

---

# Validation Splitting

If no validation CSV is provided, ARABESQ automatically creates a validation split from the training set.

Cluster-aware splitting is supported to reduce sequence homology leakage:

```bash
python -m arabesq.cli train \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir runs/example \
  --val_frac 0.1 \
  --val_split_mode cluster_stratified \
  --group_col cluster
```

---

# Model Design

ARABESQ combines:

- relation-specific graph attention encoders
- semantic-level attention fusion
- confidence-aware edge weighting
- evidence-tier integration

## Confidence-aware attention

Edge confidence scores are injected directly into attention logits:

e′ᵢⱼ = eᵢⱼ + log(sᵢⱼ + ε)

where:

- eᵢⱼ is the raw attention score
- sᵢⱼ ∈ [0,1] is the edge confidence
- low-confidence edges are progressively suppressed during message passing

---

# Output Modes

ARABESQ supports three prediction modes:

| Mode | Description |
|---|---|
| `multilabel` | Independent organelle probabilities |
| `softmax4` | Four-class mutually exclusive prediction |
| `dirichlet4` | Evidential uncertainty-aware prediction |

---

# Generated Outputs

Typical outputs include:

```text
metrics_overall.json
metrics_per_class.csv
confusion.png
graph_stats.json
semantic_attention.json
test_predictions.csv
misclassified.csv
```

---

# Reproducibility Notebook

The repository includes a paper-oriented reproducibility notebook:

```text
notebooks/ARABESQ_Paper_Repro.ipynb
```

The notebook demonstrates:

- graph ablation benchmarking
- calibration workflows
- prediction pipelines
- confusion matrix generation
- graph statistics analysis

---

# Notes

- GPU inference is strongly recommended for ESM-2 embedding extraction.
- Large embeddings, checkpoints, and generated runs should typically be excluded via `.gitignore`.
- Strict leave-one-out evaluation is supported but computationally expensive.

---

# Citation

If you use ARABESQ in your work, please cite:

```bibtex
@article{arabesq2026,
  title={ARABESQ: semantic graph learning for identifying dual-targeted proteins in Arabidopsis},
  author={...},
  journal={...},
  year={2026}
}
```