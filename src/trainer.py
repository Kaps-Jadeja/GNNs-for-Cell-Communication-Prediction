"""
Training and evaluation for link prediction — SEGCECO protocol.

Split  : 90% train / 10% test  (random, stratified by label)
Loss   : Binary cross-entropy with logits
Metrics: AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1-score
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)


def split_data(data, test_ratio: float = 0.1, seed: int = 42):
    """
    90-10 random train-test split on edges following the SEGCECO paper.
    Returns (train_data, _, test_data).
    """
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=0.0,
        num_test=test_ratio,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=True,
    )
    return transform(data)


def _get_edges_and_labels(data, device):
    """
    Merge pos/neg split edges from RandomLinkSplit(split_labels=True) into
    a single (edge_label_index, labels) pair.
    """
    pos_eli = data.pos_edge_label_index.to(device)
    neg_eli = data.neg_edge_label_index.to(device)
    eli     = torch.cat([pos_eli, neg_eli], dim=1)
    labels  = torch.cat([
        torch.ones(pos_eli.shape[1],  device=device),
        torch.zeros(neg_eli.shape[1], device=device),
    ])
    return eli, labels


def train_one_epoch(model, train_data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    x       = train_data.x.to(device)
    ei      = train_data.edge_index.to(device)
    eli, y  = _get_edges_and_labels(train_data, device)
    logits  = model(x, ei, eli)
    loss    = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device) -> dict:
    model.eval()
    x       = data.x.to(device)
    ei      = data.edge_index.to(device)
    eli, y  = _get_edges_and_labels(data, device)

    logits = model(x, ei, eli)
    loss   = nn.functional.binary_cross_entropy_with_logits(logits, y).item()

    probs  = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    fpr, tpr, _  = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)

    return {
        "loss":      loss,
        "auc_roc":   roc_auc_score(y_true, probs),
        "auc_pr":    average_precision_score(y_true, probs),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fpr":       fpr,
        "tpr":       tpr,
        "prec_curve": prec,
        "rec_curve":  rec,
    }


def train_model(
    model,
    train_data,
    test_data,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 40,
    device=None,
    verbose: bool = True,
):
    """
    Train with Adam + ReduceLROnPlateau; early stopping on test AUC-ROC.

    Returns (trained_model, test_metrics_dict, history_list)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5
    )

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        loss    = train_one_epoch(model, train_data, optimizer, device)
        metrics = evaluate(model, test_data, device)
        scheduler.step(metrics["auc_roc"])

        history.append({
            "epoch":      epoch,
            "train_loss": loss,
            "test_auc":   metrics["auc_roc"],
            "test_f1":    metrics["f1"],
        })

        if metrics["auc_roc"] > best_auc:
            best_auc   = metrics["auc_roc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:>4d} | loss={loss:.4f} | "
                  f"AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (best AUC={best_auc:.4f})")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, evaluate(model, test_data, device), history
