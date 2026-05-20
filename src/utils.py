"""
utils.py
--------
Shared utilities: metric computation, plot generation, file I/O helpers.
All plots are saved to /results and returned as Figure objects for GUI embedding.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for headless saving
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Classification Metrics ────────────────────────────────────────────────────
def evaluate_classifier(y_true, y_pred, label_names=None):
    """Compute and return classification metrics as a dict."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    return metrics, report


# ── Regression Metrics ────────────────────────────────────────────────────────
def evaluate_regression(y_true, y_pred):
    """Compute and return regression metrics as a dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ── Clustering Metrics ────────────────────────────────────────────────────────
def evaluate_clustering(X, labels):
    """Return silhouette score for cluster quality."""
    score = silhouette_score(X, labels)
    return {"silhouette_score": score}


# ── Plot: Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(importances, feature_names, save=True):
    """Bar chart of Decision Tree feature importances."""
    idx    = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in idx]
    sorted_values = importances[idx]

    colors = plt.cm.YlGn(np.linspace(0.4, 0.9, len(sorted_names)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_values[::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Decision Tree – Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(sorted_values) * 1.15)

    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Saved: {path}")
    return fig


# ── Plot: Cluster Scatter ─────────────────────────────────────────────────────
def plot_clusters(X_pca, labels, n_clusters, save=True):
    """2-D PCA scatter plot coloured by cluster label."""
    cmap   = plt.cm.get_cmap("Set2", n_clusters)
    fig, ax = plt.subplots(figsize=(7, 5))

    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=18, alpha=0.7, color=cmap(k), label=f"Soil Zone {k+1}")

    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.set_title("KMeans Clustering – Soil Segmentation (PCA)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, markerscale=1.5)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "cluster_plot.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Saved: {path}")
    return fig


# ── Plot: Regression Residuals ────────────────────────────────────────────────
def plot_residuals(y_true, y_pred, save=True):
    """Residual scatter plot with zero-line and histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Scatter: predicted vs residual
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=12, color="#2E86AB")
    axes[0].axhline(0, color="#E84855", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Yield (kg/ha)", fontsize=11)
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].set_title("Residual Plot", fontsize=12, fontweight="bold")

    # Histogram of residuals
    axes[1].hist(residuals, bins=30, color="#3BB273", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="#E84855", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual Value", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")

    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "residual_plot.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Saved: {path}")
    return fig


# ── Text report savers ────────────────────────────────────────────────────────
def save_accuracy_report(metrics, report_text, label_names):
    path = os.path.join(RESULTS_DIR, "accuracy_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  DECISION TREE CLASSIFIER – ACCURACY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision : {metrics['precision']:.4f}\n")
        f.write(f"  Recall    : {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score  : {metrics['f1']:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(report_text)
    print(f"[Utils] Saved: {path}")


def save_regression_metrics(metrics):
    path = os.path.join(RESULTS_DIR, "regression_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  LINEAR REGRESSION – YIELD PREDICTION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  RMSE : {metrics['RMSE']:.4f} kg/ha\n")
        f.write(f"  MAE  : {metrics['MAE']:.4f} kg/ha\n")
        f.write(f"  R²   : {metrics['R2']:.4f}\n")
    print(f"[Utils] Saved: {path}")


def save_clustering_metrics(metrics, n_clusters):
    path = os.path.join(RESULTS_DIR, "clustering_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  KMEANS CLUSTERING – SOIL SEGMENTATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Number of Clusters : {n_clusters}\n")
        f.write(f"  Silhouette Score   : {metrics['silhouette_score']:.4f}\n")
        f.write("\n  Interpretation:\n")
        f.write("    0.71–1.00  → Strong cluster structure\n")
        f.write("    0.51–0.70  → Reasonable structure\n")
        f.write("    0.26–0.50  → Weak structure\n")
        f.write("    < 0.25    → No substantial structure\n")
    print(f"[Utils] Saved: {path}")
