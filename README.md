# GNNs for Cell-Cell Communication Prediction

**COMP-4740: Advanced Topics in Artificial Intelligence II — Winter 2026**

A systematic comparison of Graph Neural Network architectures for predicting cell-cell communication interactions from single-cell RNA-sequencing data, following the experimental protocol of the SEGCECO reference paper.

---

## Overview

Cell-cell communication (CCC) is fundamental to tissue organization and disease. This project frames CCC prediction as a **link prediction** problem on a cell-cell communication network (CCN):

- **Nodes** — individual cells, with gene expression profiles as features
- **Positive edges** — known cell-cell communication links (SoptSC-derived CCN)
- **Task** — predict which cell pairs communicate, given partial edge knowledge

Three GNN architectures are implemented and compared across **7 datasets** spanning two species and two tissue types.

---

## Models

| Model | Description |
|---|---|
| **Baseline GNN** | Simple mean neighbourhood aggregation with MLP decoder |
| **GCN** | Spectral graph convolution with symmetric normalisation (Kipf & Welling, 2017) |
| **GAT** | Multi-head attention — learns adaptive edge weights per neighbour (Veličković et al., 2018) |

All models share the same MLP link decoder operating on concatenated node embedding pairs.

---

## Datasets

### Primary — Human Pancreas (SEGCECO)
Baron et al. 2016 human pancreas scRNA-seq (GEO: GSE84133).  
Pre-processed edgelists and 300 Information-Gain-selected gene features from the [SEGCECO GitHub repository](https://github.com/sheenahora/SEGCECO).

| Dataset | Cells | CCN Edges |
|---|---|---|
| HumanD1 | 1,930 | 67,882 |
| HumanD2 | 1,724 | 60,446 |
| HumanD3 | 3,597 | 125,700 |
| HumanD4 | 1,282 | 45,458 |

### Secondary A — Mouse Pancreas (Cross-Species Validation)
Baron et al. 2016 **mouse** pancreas — identical SoptSC pipeline, different species.

| Dataset | Cells | CCN Edges |
|---|---|---|
| MouseD1 | 821 | 29,548 |
| MouseD2 | 1,061 | 37,582 |

### Secondary B — PBMC (Cross-Tissue Validation, LIANA+)
**Kang et al. 2018** (GEO: GSE96583) — ~25k PBMCs from 8 lupus patients.  
Used as the tutorial dataset in [LIANA+](https://github.com/saezlab/liana-py), the leading Python CCC framework.  
Cell types: CD4T, CD8T, B, NK, CD14 Monocytes, FCGR3A Monocytes, Dendritic cells, Megakaryocytes.  
Graph: kNN cosine similarity (n=15) on PCA embeddings, mirroring SoptSC's CCN construction.

| Dataset | Cells | Cell Types |
|---|---|---|
| KangPBMC | ~2,400 | 8 immune |

---

## Experimental Protocol

Follows the **SEGCECO paper** (Hora et al., *Briefings in Bioinformatics*, 2024):

- **Split** — 90% train / 10% test (random edge split)
- **Negative sampling** — equal number of randomly sampled non-existing cell pairs
- **Metrics** — AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1-score
- **Node features** — top 300 genes by mutual information (SelectKBest)
- **Training** — Adam optimiser, ReduceLROnPlateau scheduler, early stopping

---

## Project Structure

```
├── main.ipynb                  # Full pipeline — run this
├── requirements.txt
└── src/
    ├── data_loader.py          # Downloads & loads SEGCECO human pancreas data
    ├── secondary_datasets.py   # Mouse pancreas + Kang 2018 PBMC (LIANA+)
    ├── graph_builder.py        # Constructs PyG Data for link prediction
    ├── models.py               # Baseline GNN, GCN, GAT architectures
    ├── trainer.py              # Training loop, evaluation, 90-10 split
    ├── utils.py                # Metrics, plots, seeding
    └── lr_database.py          # Human LR pair database (~170 pairs, 30+ pathways)
```

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.5+, PyTorch Geometric 2.7+

```bash
pip install torch torch-geometric
pip install scanpy anndata
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

---

## Running

```bash
jupyter notebook main.ipynb
```

The notebook is self-contained:
- **Data downloads automatically** from [SEGCECO GitHub](https://github.com/sheenahora/SEGCECO) and [figshare](https://figshare.com/ndownloader/files/34464122) on first run, then cached in `data/`
- **No manual data preparation** required
- Run all cells top-to-bottom to reproduce all results and figures

---

## References

- **SEGCECO**: Hora S. et al. *SEGCECO: subgraph-based explainable graph convolutional network for predicting cell–cell communication*. Briefings in Bioinformatics, 2024. https://doi.org/10.1093/bib/bbae160
- **Baron et al.**: Baron M. et al. *A Single-Cell Transcriptomic Map of the Human and Mouse Pancreas*. Cell Systems, 2016. GEO: GSE84133
- **Kang et al.**: Kang H. et al. *Multiplexed droplet single-cell RNA-sequencing using natural genetic variation*. Nature Biotechnology, 2018. GEO: GSE96583
- **LIANA+**: Dimitrov D. et al. *LIANA+ provides an all-in-one framework for cell–cell communication inference*. Nature Cell Biology, 2024. https://doi.org/10.1038/s41556-024-01469-w
- **GCN**: Kipf T.N. & Welling M. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR, 2017.
- **GAT**: Veličković P. et al. *Graph Attention Networks*. ICLR, 2018.
