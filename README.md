# GNNs for Cell-Cell Communication Prediction


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

## Results

### Average across all 7 datasets

| Model | AUC-ROC | AUC-PR | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **GCN** | **0.9887** | **0.9843** | **0.9644** | **0.9440** | **0.9878** | **0.9653** |
| GAT | 0.9819 | 0.9745 | 0.9502 | 0.9256 | 0.9796 | 0.9517 |
| Baseline GNN | 0.9722 | 0.9601 | 0.9359 | 0.8993 | 0.9824 | 0.9389 |

GCN is the top performer across every metric. GAT is a close second; the Baseline GNN remains competitive given its simplicity.

### Per-dataset breakdown

| Dataset | Model | AUC-ROC | AUC-PR | Accuracy | F1 |
|---|---|---|---|---|---|
| HumanD1 | GCN | 0.9915 | 0.9879 | 0.9708 | 0.9716 |
| HumanD1 | GAT | 0.9816 | 0.9745 | 0.9512 | 0.9530 |
| HumanD1 | Baseline | 0.9741 | 0.9634 | 0.9383 | 0.9414 |
| HumanD2 | GCN | 0.9948 | 0.9919 | 0.9796 | 0.9799 |
| HumanD2 | GAT | 0.9951 | 0.9923 | 0.9799 | 0.9802 |
| HumanD2 | Baseline | 0.9873 | 0.9803 | 0.9618 | 0.9630 |
| HumanD3 | GCN | 0.9960 | 0.9947 | 0.9791 | 0.9794 |
| HumanD3 | GAT | 0.9859 | 0.9769 | 0.9549 | 0.9565 |
| HumanD3 | Baseline | 0.9620 | 0.9427 | 0.9183 | 0.9233 |
| HumanD4 | GCN | 0.9751 | 0.9628 | 0.9424 | 0.9448 |
| HumanD4 | GAT | 0.9731 | 0.9596 | 0.9383 | 0.9410 |
| HumanD4 | Baseline | 0.9691 | 0.9527 | 0.9322 | 0.9357 |
| MouseD1 | GCN | 0.9886 | 0.9854 | 0.9601 | 0.9610 |
| MouseD1 | GAT | 0.9679 | 0.9575 | 0.9193 | 0.9219 |
| MouseD1 | Baseline | 0.9591 | 0.9434 | 0.9115 | 0.9159 |
| MouseD2 | GCN | 0.9963 | 0.9947 | 0.9798 | 0.9800 |
| MouseD2 | GAT | 0.9931 | 0.9922 | 0.9690 | 0.9694 |
| MouseD2 | Baseline | 0.9877 | 0.9832 | 0.9655 | 0.9664 |
| KangPBMC | GCN | 0.9784 | 0.9726 | 0.9390 | 0.9405 |
| KangPBMC | GAT | 0.9766 | 0.9687 | 0.9385 | 0.9400 |
| KangPBMC | Baseline | 0.9662 | 0.9548 | 0.9234 | 0.9266 |

### Key observations

- **GCN generalises best** — consistently top or near-top across human pancreas, mouse pancreas, and PBMC datasets
- **Cross-species transfer holds** — mouse pancreas results (MouseD1/2) closely mirror human pancreas, confirming the CCN structure is species-agnostic
- **Cross-tissue transfer is harder** — KangPBMC scores are ~2–3% lower in F1 vs. pancreas datasets, reflecting the coarser kNN-based graph vs. SoptSC-derived CCN
- **GAT ≈ GCN on HumanD2** — the only dataset where attention-based weighting fully matches spectral convolution
- **Baseline GNN is surprisingly strong** (F1 > 0.92 everywhere), suggesting the node features alone carry most of the predictive signal

---

## References

- **SEGCECO**: Hora S. et al. *SEGCECO: subgraph-based explainable graph convolutional network for predicting cell–cell communication*. Briefings in Bioinformatics, 2024. https://doi.org/10.1093/bib/bbae160
- **Baron et al.**: Baron M. et al. *A Single-Cell Transcriptomic Map of the Human and Mouse Pancreas*. Cell Systems, 2016. GEO: GSE84133
- **Kang et al.**: Kang H. et al. *Multiplexed droplet single-cell RNA-sequencing using natural genetic variation*. Nature Biotechnology, 2018. GEO: GSE96583
- **LIANA+**: Dimitrov D. et al. *LIANA+ provides an all-in-one framework for cell–cell communication inference*. Nature Cell Biology, 2024. https://doi.org/10.1038/s41556-024-01469-w
- **GCN**: Kipf T.N. & Welling M. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR, 2017.
- **GAT**: Veličković P. et al. *Graph Attention Networks*. ICLR, 2018.
