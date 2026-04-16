"""
Graph builder for link prediction.

Passes the CCN Data object through unchanged — RandomLinkSplit in
trainer.py handles the 90-10 split and balanced negative sampling
following the SEGCECO protocol.
"""

from torch_geometric.data import Data


def build_link_prediction_data(data: Data, seed: int = 42) -> Data:
    """
    Returns a copy of `data` with identical fields.
    The edge_index holds the positive CCN edges; RandomLinkSplit will:
      - split them 90-10 into train / test positive edges
      - add an equal number of random negative edges to each split

    This matches SEGCECO: '10% of links as testing, 90% as training,
    with an equal number of randomly chosen unconnected pairs as negatives.'
    """
    n_pos = data.edge_index.shape[1]
    name  = getattr(data, "dataset_name", "unknown")
    print(f"[graph_builder] {name} | {data.num_nodes} nodes | "
          f"{n_pos} positive CCN edges")
    return data
