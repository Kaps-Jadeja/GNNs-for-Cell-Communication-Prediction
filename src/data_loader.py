"""
Data loader for SEGCECO human pancreas cell-cell communication datasets.

Downloads pre-processed files directly from the SEGCECO GitHub repository:
  https://github.com/sheenahora/SEGCECO

For each dataset (HumanD1 – HumanD4) the following files are used:
  - edgelist_encoded_HumanDX.txt   : cell-cell communication edges
  - attributes_IG_HumanDX.csv      : node features (IG-selected gene expression)
  - 300_feature_selected_ig.csv    : ranked gene list (top-300 by mutual info)

All files match the exact preprocessing pipeline of SEGCECO:
  normalize_total(30 000) → log1p → HVG (min_disp=0.5) →
  top-300 genes by mutual information (SelectKBest)
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ──────────────────────────────────────────────────────────────────────────────
# GitHub raw base URL for SEGCECO data
# ──────────────────────────────────────────────────────────────────────────────
_BASE = "https://raw.githubusercontent.com/sheenahora/SEGCECO/master/data"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASETS = {
    "HumanD1": {
        "edgelist":   f"{_BASE}/HumanD1/edgelist_encoded_HumanD1.txt",
        "attributes": f"{_BASE}/HumanD1/attributes_IG_HumanD1.csv",
        "features":   f"{_BASE}/HumanD1/300_feature_selected_ig.csv",
    },
    "HumanD2": {
        "edgelist":   f"{_BASE}/HumanD2/edgelist_encoded_HumanD2.txt",
        "attributes": f"{_BASE}/HumanD2/attributes_IG_HumanD2.csv",
        "features":   f"{_BASE}/HumanD2/300_feature_selected_ig.csv",
    },
    "HumanD3": {
        "edgelist":   f"{_BASE}/HumanD3/edgelist_encoded_HumanD3.txt",
        "attributes": f"{_BASE}/HumanD3/attributes_IG_HumanD3.csv",
        "features":   f"{_BASE}/HumanD3/300_feature_selected_ig.csv",
    },
    "HumanD4": {
        "edgelist":   f"{_BASE}/HumanD4/edgelist_encoded_HumanD4.txt",
        "attributes": f"{_BASE}/HumanD4/attributes_IG_HumanD4.csv",
        "features":   f"{_BASE}/HumanD4/300_feature_selected_ig.csv",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str = "HumanD1", n_features: int = 300) -> Data:
    """
    Load one SEGCECO human pancreas dataset as a PyG Data object.

    Parameters
    ----------
    name       : One of 'HumanD1', 'HumanD2', 'HumanD3', 'HumanD4'.
    n_features : Number of top IG-selected genes to use as node features
                 (max 300; use None to keep all columns in attributes file).

    Returns
    -------
    PyG Data with:
        data.x              : (N, F) float node features (gene expression)
        data.edge_index     : (2, E) all positive (CCN) edges
        data.y              : (N,)   integer cell-type labels
        data.num_nodes      : N
        data.dataset_name   : str
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}")

    urls      = DATASETS[name]
    local_dir = os.path.join(DATA_DIR, name)
    os.makedirs(local_dir, exist_ok=True)

    edgelist_path   = _download(urls["edgelist"],   local_dir, f"edgelist_{name}.txt")
    attributes_path = _download(urls["attributes"], local_dir, f"attributes_{name}.csv")
    features_path   = _download(urls["features"],   local_dir, f"features300_{name}.csv")

    edges, node_map = _parse_edgelist(edgelist_path)
    attrs, labels   = _parse_attributes(attributes_path, node_map, features_path, n_features)

    n_nodes = len(node_map)
    x       = torch.tensor(attrs, dtype=torch.float)
    y       = torch.tensor(labels, dtype=torch.long)
    ei      = torch.tensor(edges.T, dtype=torch.long)   # (2, E)

    data = Data(x=x, edge_index=ei, y=y, num_nodes=n_nodes)
    data.dataset_name = name

    print(f"[{name}] nodes={n_nodes}, edges={ei.shape[1]}, "
          f"features={x.shape[1]}, cell-types={len(set(labels.tolist()))}")
    return data


def load_all_human_datasets(n_features: int = 300) -> dict:
    """Load all four human pancreas datasets. Returns {name: Data}."""
    return {name: load_dataset(name, n_features) for name in DATASETS}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download(url: str, local_dir: str, filename: str) -> str:
    """Download file if not already cached; return local path."""
    path = os.path.join(local_dir, filename)
    if os.path.exists(path):
        return path
    print(f"  Downloading {filename} …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
        print("done")
    except Exception as e:
        # Try .csv variant for edgelists
        if filename.endswith(".txt"):
            csv_url = url.replace(".txt", ".csv")
            try:
                urllib.request.urlretrieve(csv_url, path)
                print("done (csv)")
                return path
            except Exception:
                pass
        raise RuntimeError(f"Failed to download {url}: {e}")
    return path


def _parse_edgelist(path: str) -> tuple[np.ndarray, dict]:
    """
    Parse a space/comma-separated edgelist file.
    Returns:
        edges    : (E, 2) int array of re-indexed (0-based) edges
        node_map : {original_id -> new_index}
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    rows.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue   # skip header if present

    if not rows:
        raise ValueError(f"No valid edges found in {path}")

    # Build consecutive node ID mapping
    unique_nodes = sorted(set(n for edge in rows for n in edge))
    node_map     = {orig: idx for idx, orig in enumerate(unique_nodes)}

    edges = np.array([[node_map[s], node_map[d]] for s, d in rows], dtype=np.int64)
    return edges, node_map


def _parse_attributes(
    attr_path: str,
    node_map: dict,
    feat_path: str,
    n_features: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the attributes CSV and align rows to node_map order.

    The SEGCECO attributes_IG file contains ONLY gene expression features
    (all columns are selected genes — no separate cell-type label column).
    Rows correspond to individual cells; row order matches the original
    cell indexing used in the edgelist.

    Returns
    -------
    features : (N, F) float array, rows ordered by node_map index
    labels   : (N,)  int array — zeros (cell-type labels not in this file)
    """
    df = pd.read_csv(attr_path)

    # ── All columns are gene expression features ─────────────────────────────
    # Optionally restrict to the top-n ranked genes
    if n_features is not None and os.path.exists(feat_path):
        feat_df   = pd.read_csv(feat_path)
        feat_col  = "Feature_Name" if "Feature_Name" in feat_df.columns else feat_df.columns[0]
        top_genes = feat_df[feat_col].iloc[:n_features].tolist()
        available = [g for g in top_genes if g in df.columns]
        if available:
            df = df[available]

    feat_arr  = df.values.astype(np.float32)
    n_nodes   = len(node_map)
    label_arr = np.zeros(n_nodes, dtype=np.int64)   # placeholder; not used in training

    # ── Align rows to node_map ───────────────────────────────────────────────
    if feat_arr.shape[0] == n_nodes:
        return feat_arr, label_arr

    # More rows than graph nodes: select only cells present in the CCN
    ordered_feats = np.zeros((n_nodes, feat_arr.shape[1]), dtype=np.float32)
    for orig_id, new_idx in node_map.items():
        if orig_id < feat_arr.shape[0]:
            ordered_feats[new_idx] = feat_arr[orig_id]

    return ordered_feats, label_arr
