"""
Publication-quality visualization for swallowing manifold analysis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Dict, Optional

from ..core.trajectory import SwallowingTrajectory


PHENOTYPE_COLORS = {
    "healthy": "#2ecc71", "fibrotic": "#e74c3c", "weak": "#3498db",
    "compensatory": "#f39c12", "neurogenic": "#9b59b6",
}
PHASE_COLORS = ["#1abc9c", "#e74c3c", "#9b59b6", "#f39c12", "#3498db"]


def plot_trajectories_3d(
    trajectories: List[SwallowingTrajectory],
    n_per_condition: int = 8,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Plot trajectories in 3D PCA embedding, colored by condition."""
    X = np.vstack([t.landmarks for t in trajectories])
    pca = PCA(n_components=3)
    pca.fit(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    conditions = sorted(set(t.condition for t in trajectories))
    for cond in conditions:
        cond_trajs = [t for t in trajectories if t.condition == cond]
        color = PHENOTYPE_COLORS.get(cond, "#7f8c8d")
        for traj in cond_trajs[:n_per_condition]:
            emb = pca.transform(traj.landmarks)
            ax.plot(emb[:, 0], emb[:, 1], emb[:, 2],
                    color=color, alpha=0.5, linewidth=0.8)
        ax.plot([], [], [], color=color, linewidth=2.5, label=cond.capitalize())

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Swallowing Trajectories in Manifold Embedding")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_scree(
    X: np.ndarray,
    true_dim: int = 5,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Plot PCA scree and cumulative variance."""
    pca = PCA(n_components=min(20, X.shape[1]))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.bar(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, color="#3498db", alpha=0.7)
    ax1.axvline(true_dim, color="red", linestyle="--", alpha=0.7, label=f"d = {true_dim}")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("(a) Individual Component Variance")
    ax1.legend()

    ax2.plot(range(1, len(cumvar)+1), cumvar, "o-", color="#e74c3c", markersize=5)
    ax2.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
    ax2.axhline(0.99, color="gray", linestyle="--", alpha=0.5, label="99%")
    ax2.axvline(true_dim, color="red", linestyle="--", alpha=0.7, label=f"d = {true_dim}")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("(b) Cumulative Explained Variance")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric_boxplots(
    df,
    metric_col: str,
    ylabel: str = "",
    title: str = "",
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Box plot of a metric across phenotypes."""
    conditions = ["fibrotic", "healthy", "weak", "compensatory", "neurogenic"]
    conditions = [c for c in conditions if c in df["condition"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x="condition", y=metric_col, order=conditions,
                palette=PHENOTYPE_COLORS, width=0.6, ax=ax)
    sns.stripplot(data=df, x="condition", y=metric_col, order=conditions,
                  color="black", alpha=0.25, size=3, ax=ax)
    ax.set_xlabel("Dysphagia Phenotype")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title or metric_col)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
