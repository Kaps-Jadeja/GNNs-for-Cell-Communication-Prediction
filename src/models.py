"""
Three GNN architectures for cell-cell communication link prediction.

All models share the same interface:
    encode(x, edge_index)  -> node embeddings (N, hidden_dim)
    decode(z, edge_index)  -> edge logits     (E,)
    forward(x, edge_index, edge_label_index) -> edge logits (E,)

Link decoder: MLP on concatenated (z_src ∥ z_dst).

Input dimension is typically 300 (IG-selected genes from SEGCECO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops


# ──────────────────────────────────────────────────────────────────────────────
# Shared MLP link decoder
# ──────────────────────────────────────────────────────────────────────────────

class LinkDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        return self.net(torch.cat([z[src], z[dst]], dim=-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Model 1 – Baseline GNN  (mean neighbourhood aggregation + MLP)
# ──────────────────────────────────────────────────────────────────────────────

class _MeanAggLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n  = x.size(0)
        ei, _ = add_self_loops(edge_index, num_nodes=n)
        src, dst = ei[0], ei[1]
        agg = torch.zeros(n, x.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        deg = torch.bincount(dst, minlength=n).float().clamp(min=1).unsqueeze(1)
        return F.relu(self.lin(agg / deg))


class BaselineGNN(nn.Module):
    """
    Baseline: two mean-aggregation layers followed by an MLP link decoder.
    Represents the simplest possible graph-based approach.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers  = nn.ModuleList(
            [_MeanAggLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.decoder = LinkDecoder(hidden_dim)

    def encode(self, x, edge_index):
        for layer in self.layers:
            x = F.dropout(layer(x, edge_index), p=self.dropout, training=self.training)
        return x

    def decode(self, z, eli):
        return self.decoder(z, eli)

    def forward(self, x, edge_index, edge_label_index):
        return self.decode(self.encode(x, edge_index), edge_label_index)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2 – GCN  (Kipf & Welling 2017)
# ──────────────────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """
    Two GCNConv layers with residual input projection and LayerNorm.
    Symmetric normalisation: Ã = D^{-½} A D^{-½}
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.decoder = LinkDecoder(hidden_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(F.dropout(x, p=self.dropout, training=self.training), edge_index)))
        return x

    def decode(self, z, eli):
        return self.decoder(z, eli)

    def forward(self, x, edge_index, edge_label_index):
        return self.decode(self.encode(x, edge_index), edge_label_index)


# ──────────────────────────────────────────────────────────────────────────────
# Model 3 – GAT  (Veličković et al. 2018)
# ──────────────────────────────────────────────────────────────────────────────

class GAT(nn.Module):
    """
    Two GATConv layers with multi-head attention.
    Attention enables the model to learn cell-type-specific interaction weights,
    going beyond uniform neighbourhood aggregation used by GCN/Baseline.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        head_out = max(1, hidden_dim // heads)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, head_out, heads=heads, concat=True,
                    dropout=dropout, add_self_loops=True)
            for _ in range(n_layers)
        ])
        self.norms    = nn.ModuleList([nn.LayerNorm(head_out * heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(head_out * heads, hidden_dim)
        self.decoder  = LinkDecoder(hidden_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.elu(conv(F.dropout(x, p=self.dropout, training=self.training), edge_index)))
        return self.out_proj(x)

    def decode(self, z, eli):
        return self.decoder(z, eli)

    def forward(self, x, edge_index, edge_label_index):
        return self.decode(self.encode(x, edge_index), edge_label_index)
