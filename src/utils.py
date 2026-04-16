"""
Utility: seeding, metric printing, and result visualisation.
"""

import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_metrics(name: str, m: dict):
    print(f"\n{'═'*56}")
    print(f"  {name}")
    print(f"{'─'*56}")
    print(f"  AUC-ROC   : {m['auc_roc']:.4f}")
    print(f"  AUC-PR    : {m['auc_pr']:.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"{'═'*56}")


def results_table(results: dict):
    """Print a compact table for the report."""
    print(f"\n{'Model':<20} {'AUC-ROC':>9} {'AUC-PR':>9} {'Acc':>8} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("─" * 74)
    for name, m in results.items():
        print(f"{name:<20} {m['auc_roc']:>9.4f} {m['auc_pr']:>9.4f} "
              f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}")


def plot_roc_curves(results: dict, title: str = "", save: bool = True):
    plt.figure(figsize=(6, 5))
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    for (name, m), c in zip(results.items(), colors):
        plt.plot(m["fpr"], m["tpr"], label=f"{name} ({m['auc_roc']:.3f})", color=c, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves {title}"); plt.legend(loc="lower right")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"roc_{title.replace(' ','_')}.png"), dpi=150)
    plt.show()


def plot_pr_curves(results: dict, title: str = "", save: bool = True):
    plt.figure(figsize=(6, 5))
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    for (name, m), c in zip(results.items(), colors):
        plt.plot(m["rec_curve"], m["prec_curve"],
                 label=f"{name} ({m['auc_pr']:.3f})", color=c, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves {title}"); plt.legend(loc="upper right")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"pr_{title.replace(' ','_')}.png"), dpi=150)
    plt.show()


def plot_training_history(histories: dict, title: str = "", save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    for (name, hist), c in zip(histories.items(), colors):
        ep  = [h["epoch"]      for h in hist]
        lss = [h["train_loss"] for h in hist]
        auc = [h["test_auc"]   for h in hist]
        axes[0].plot(ep, lss, label=name, color=c, lw=1.5)
        axes[1].plot(ep, auc, label=name, color=c, lw=1.5)
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss",  title=f"Training Loss {title}")
    axes[1].set(xlabel="Epoch", ylabel="AUC-ROC",   title=f"Test AUC-ROC {title}")
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"history_{title.replace(' ','_')}.png"), dpi=150)
    plt.show()


def plot_metric_comparison(results: dict, title: str = "", save: bool = True):
    """Bar chart: AUC-ROC, AUC-PR, F1 side-by-side for each model."""
    models  = list(results.keys())
    metrics = ["auc_roc", "auc_pr", "f1"]
    labels  = ["AUC-ROC", "AUC-PR", "F1"]
    colors  = ["#3498DB", "#E74C3C", "#2ECC71"]
    x, w    = np.arange(len(models)), 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (met, lab, col) in enumerate(zip(metrics, labels, colors)):
        ax.bar(x + i * w, [results[m][met] for m in models], w,
               label=lab, color=col, alpha=0.85)
    ax.set_xticks(x + w); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison {title}"); ax.legend()
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"comparison_{title.replace(' ','_')}.png"), dpi=150)
    plt.show()


def plot_cross_dataset_heatmap(all_results: dict, metric: str = "auc_roc", save: bool = True):
    """
    Heatmap of a chosen metric across all (dataset, model) combinations.
    all_results : {dataset_name: {model_name: metrics_dict}}
    """
    datasets = list(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    matrix   = np.array(
        [[all_results[d][m][metric] for m in models] for d in datasets]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(models) + 1), max(4, len(datasets))))
    sns.heatmap(
        matrix, xticklabels=models, yticklabels=datasets,
        annot=True, fmt=".3f", cmap="YlGn",
        vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
    )
    ax.set_title(f"{metric.upper()} across datasets and models")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"heatmap_{metric}.png"), dpi=150)
    plt.show()
