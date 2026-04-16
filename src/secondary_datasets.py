"""
Secondary datasets for cell-cell communication link prediction.

1. SEGCECO Mouse Pancreas  (MouseD1, MouseD2)
   Same SoptSC-derived CCN format as the primary human pancreas datasets.
   Source: https://github.com/sheenahora/SEGCECO

2. Kang et al. 2018 PBMC  (LIANA+ tutorial dataset)
   ~25k PBMCs from 8 lupus patients, 8 immune cell types.
   Source: LIANA+ / pertpy  https://figshare.com/ndownloader/files/34464122
   Reference: Kang et al., Nat Biotechnol 36, 89-94 (2018). GSE96583.
   Graph built with scanpy kNN (cosine similarity, n_neighbors=15) to
   approximate the SoptSC CCN construction used in SEGCECO.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ──────────────────────────────────────────────────────────────────────────────
# Paths / URLs
# ──────────────────────────────────────────────────────────────────────────────
_SEGCECO_BASE = "https://raw.githubusercontent.com/sheenahora/SEGCECO/master/data"
DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")

MOUSE_DATASETS = {
    "MouseD1": {
        "edgelist":   f"{_SEGCECO_BASE}/MouseD1/edgelist_encoded_MouseD1.txt",
        "attributes": f"{_SEGCECO_BASE}/MouseD1/attributes_IG_MouseD1.csv",
        "features":   f"{_SEGCECO_BASE}/MouseD1/300_feature_selected_ig.csv",
    },
    "MouseD2": {
        "edgelist":   f"{_SEGCECO_BASE}/MouseD2/edgelist_encoded_MouseD2.txt",
        "attributes": f"{_SEGCECO_BASE}/MouseD2/attributes_IG_MouseD2.csv",
        "features":   f"{_SEGCECO_BASE}/MouseD2/300_feature_selected_ig.csv",
    },
}

KANG_URL  = "https://figshare.com/ndownloader/files/34464122"
KANG_H5AD = os.path.join(DATA_DIR, "kang", "kang_counts_25k.h5ad")


# ──────────────────────────────────────────────────────────────────────────────
# Mouse pancreas (SEGCECO format — reuses data_loader helpers)
# ──────────────────────────────────────────────────────────────────────────────

def load_mouse_dataset(name: str = "MouseD1", n_features: int = 300) -> Data:
    """Load MouseD1 or MouseD2 from the SEGCECO GitHub repository."""
    from src.data_loader import _download, _parse_edgelist, _parse_attributes

    if name not in MOUSE_DATASETS:
        raise ValueError(f"Unknown mouse dataset '{name}'. Choose from {list(MOUSE_DATASETS)}")

    urls      = MOUSE_DATASETS[name]
    local_dir = os.path.join(DATA_DIR, name)
    os.makedirs(local_dir, exist_ok=True)

    edge_path  = _download(urls["edgelist"],   local_dir, f"edgelist_{name}.txt")
    attr_path  = _download(urls["attributes"], local_dir, f"attributes_{name}.csv")
    feat_path  = _download(urls["features"],   local_dir, f"features300_{name}.csv")

    edges, node_map = _parse_edgelist(edge_path)
    attrs, labels   = _parse_attributes(attr_path, node_map, feat_path, n_features)

    n_nodes = len(node_map)
    data = Data(
        x          = torch.tensor(attrs,       dtype=torch.float),
        y          = torch.tensor(labels,      dtype=torch.long),
        edge_index = torch.tensor(edges.T,     dtype=torch.long),
        num_nodes  = n_nodes,
    )
    data.dataset_name = name
    print(f"[{name}] nodes={n_nodes}, edges={data.edge_index.shape[1]}, "
          f"features={data.x.shape[1]}")
    return data


def load_all_mouse_datasets(n_features: int = 300) -> dict:
    return {name: load_mouse_dataset(name, n_features) for name in MOUSE_DATASETS}


# ──────────────────────────────────────────────────────────────────────────────
# Kang 2018 PBMC  (LIANA+ dataset)
# ──────────────────────────────────────────────────────────────────────────────

def load_kang_pbmc(
    n_cells_per_type: int = 300,
    n_features:       int = 300,
    n_neighbors:      int = 15,
    seed:             int = 42,
) -> Data:
    """
    Load and process the Kang et al. 2018 PBMC dataset (LIANA+ tutorial data).

    Steps
    -----
    1. Download h5ad from figshare (cached after first run).
    2. Subsample up to n_cells_per_type cells per cell type (for tractability).
    3. Preprocess: normalize_total(1e4) → log1p → highly_variable_genes → scale.
    4. Build kNN cell-cell similarity graph (cosine, n_neighbors=15) —
       mirrors SoptSC's CCN construction used in SEGCECO.
    5. Select top n_features genes by mutual information (IG selection).
    6. Return PyG Data (nodes = individual cells, edges = kNN CCN).

    Cell types : CD4T, CD8T, B, NK, CD14 Monocytes, FCGR3A Monocytes,
                 Dendritic cells, Megakaryocytes  (8 immune cell types).
    """
    try:
        import scanpy as sc
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
    except ImportError as e:
        raise ImportError(f"Install scanpy: pip install scanpy\n{e}")

    os.makedirs(os.path.dirname(KANG_H5AD), exist_ok=True)

    # ── 1. Download ─────────────────────────────────────────────────────────
    if not os.path.exists(KANG_H5AD):
        print("  Downloading Kang 2018 PBMC from figshare (~200 MB)...", flush=True)
        urllib.request.urlretrieve(KANG_URL, KANG_H5AD)
        print("  Download complete.")

    print("  Loading Kang 2018 PBMC dataset...")
    adata = sc.read_h5ad(KANG_H5AD)

    # ── Harmonise cell type column ───────────────────────────────────────────
    for col in ["cell_type", "celltype", "cell_abbr", "label"]:
        if col in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[col].astype(str)
            break

    # ── 2. Subsample ─────────────────────────────────────────────────────────
    np.random.seed(seed)
    keep_idx = []
    for ct in adata.obs["cell_type"].unique():
        idx = np.where(adata.obs["cell_type"] == ct)[0]
        chosen = np.random.choice(idx, min(n_cells_per_type, len(idx)), replace=False)
        keep_idx.extend(chosen.tolist())
    adata = adata[sorted(keep_idx)].copy()
    print(f"  Subsampled to {adata.n_obs} cells across "
          f"{adata.obs['cell_type'].nunique()} cell types.")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)

    # ── 4. Build kNN CCN ──────────────────────────────────────────────────────
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric="cosine",
                    n_pcs=50, random_state=seed)

    # Extract kNN graph from scanpy connectivity matrix
    import scipy.sparse as sp
    conn = adata.obsp["connectivities"]
    conn_coo = sp.coo_matrix(conn)
    src = conn_coo.row.astype(np.int64)
    dst = conn_coo.col.astype(np.int64)

    # ── 5. IG gene selection ──────────────────────────────────────────────────
    import scipy.sparse as sp2
    X = adata.X
    if sp2.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    ct_labels = pd.factorize(adata.obs["cell_type"])[0]
    n_select  = min(n_features, X.shape[1])
    selector  = SelectKBest(mutual_info_classif, k=n_select)
    X_sel     = selector.fit_transform(X, ct_labels).astype(np.float32)

    print(f"  IG feature selection: {X.shape[1]} → {X_sel.shape[1]} genes")

    # ── 6. Build PyG Data ─────────────────────────────────────────────────────
    n_nodes   = adata.n_obs
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    data = Data(
        x          = torch.tensor(X_sel,    dtype=torch.float),
        y          = torch.tensor(ct_labels, dtype=torch.long),
        edge_index = edge_index,
        num_nodes  = n_nodes,
    )
    data.dataset_name = "KangPBMC"

    print(f"[KangPBMC] nodes={n_nodes}, edges={edge_index.shape[1]}, "
          f"features={X_sel.shape[1]}, "
          f"cell-types={adata.obs['cell_type'].nunique()}")
    return data
