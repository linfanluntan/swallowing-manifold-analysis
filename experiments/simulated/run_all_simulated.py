#!/usr/bin/env python3
"""
Master script: run all simulated experiments.

Validates each theoretical prediction from the paper using synthetic
swallowing trajectories with known ground truth.

Usage:
    python experiments/simulated/run_all_simulated.py --config configs/simulation_config.yaml
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.manifold import SwallowingManifold
from core.trajectory import SwallowingTrajectory
from core.metrics import extract_all_metrics, geodesic_length, mean_curvature
from core.srvf import (
    srvf_distance_matrix,
    time_warp_invariance_test,
    srvf_distance_with_alignment,
)
from core.phase_detection import detect_phases_geometric, phase_metrics, bottleneck_traversal_score
from core.phenotype import PhenotypeClassifier
from simulation.trajectory_generator import (
    SyntheticManifold,
    TrajectoryGenerator,
    DEFAULT_LANDMARK_GROUPS,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["tables_dir"], exist_ok=True)


# =====================================================================
# Experiment 1: Manifold dimensionality
# =====================================================================
def exp01_manifold_dim(cohort, cfg, fig_dir, tab_dir):
    """Validate that intrinsic dimension d << N is recoverable."""
    print("\n[EXP01] Manifold dimensionality estimation...")

    # Stack all landmark data
    X = np.vstack([t.landmarks for t in cohort])
    true_dim = cfg["manifold"]["intrinsic_dim"]

    manifold = SwallowingManifold(
        n_components=true_dim,
        method="isomap",
        n_neighbors=cfg["manifold_learning"]["n_neighbors"],
    )

    result = manifold.estimate_intrinsic_dimension(
        X, max_dim=cfg["manifold_learning"]["max_dim_test"]
    )

    # Also compute PCA explained variance
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(20, X.shape[1], X.shape[0]))
    pca.fit(X)

    # --- Figure: Scree plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_), "bo-", markersize=5)
    ax.axhline(0.95, color="r", linestyle="--", alpha=0.7, label="95% variance")
    ax.axvline(true_dim, color="g", linestyle="--", alpha=0.7,
               label=f"True dim (d={true_dim})")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Eigenvalue Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result["dimensions"], result["scores"], "ro-", markersize=5)
    ax.axvline(true_dim, color="g", linestyle="--", alpha=0.7,
               label=f"True dim (d={true_dim})")
    ax.axvline(result["estimated_dim"], color="b", linestyle=":",
               alpha=0.7, label=f"Estimated (d={result['estimated_dim']})")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Residual variance explained")
    ax.set_title("Isomap Residual Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp01_manifold_dim.png", dpi=cfg["output"]["dpi"])
    plt.close()

    # --- Table ---
    var_at_true = float(np.cumsum(pca.explained_variance_ratio_)[true_dim - 1])
    summary = {
        "True intrinsic dimension": true_dim,
        "Estimated dimension (Isomap)": result["estimated_dim"],
        "Ambient dimension": X.shape[1],
        "PCA variance at d=5": f"{var_at_true:.4f}",
        "N samples": X.shape[0],
    }
    pd.DataFrame([summary]).to_csv(f"{tab_dir}/exp01_manifold_dim.csv", index=False)
    print(f"  True dim={true_dim}, Estimated={result['estimated_dim']}, "
          f"PCA var@d={true_dim}: {var_at_true:.3f}")


# =====================================================================
# Experiment 2: Trajectory types
# =====================================================================
def exp02_trajectory_types(cohort, cfg, fig_dir, tab_dir):
    """Visualize healthy vs pathological trajectory geometry."""
    print("\n[EXP02] Trajectory type visualization...")

    # Embed all trajectories into 3D for visualization
    X = np.vstack([t.landmarks for t in cohort])
    manifold = SwallowingManifold(n_components=3, method="pca")
    manifold.fit(X)

    fig = plt.figure(figsize=(14, 10))
    conditions = ["healthy", "fibrotic", "weak", "compensatory", "neurogenic"]
    colors = {"healthy": "#2ecc71", "fibrotic": "#e74c3c", "weak": "#3498db",
              "compensatory": "#f39c12", "neurogenic": "#9b59b6"}

    ax = fig.add_subplot(111, projection="3d")
    for cond in conditions:
        cond_trajs = [t for t in cohort if t.condition == cond]
        # Plot first 5 trajectories per condition
        for traj in cond_trajs[:5]:
            embedded = manifold.transform(traj.landmarks)
            ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                    color=colors[cond], alpha=0.6, linewidth=1.2)
        # Dummy for legend
        ax.plot([], [], [], color=colors[cond], linewidth=2, label=cond.capitalize())

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Swallowing Trajectories in Manifold Embedding (PCA 3D)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp02_trajectory_types.png", dpi=cfg["output"]["dpi"])
    plt.close()
    print("  Saved trajectory visualization.")


# =====================================================================
# Experiment 3: Geodesic length
# =====================================================================
def exp03_geodesic_length(cohort, cfg, fig_dir, tab_dir):
    """Validate geodesic length as effort/efficiency measure."""
    print("\n[EXP03] Geodesic length analysis...")

    data = []
    for traj in cohort:
        gl = geodesic_length(traj)
        data.append({"condition": traj.condition, "geodesic_length": gl})

    df = pd.DataFrame(data)

    # --- Figure: Box plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["healthy", "weak", "fibrotic", "compensatory", "neurogenic"]
    palette = {"healthy": "#2ecc71", "fibrotic": "#e74c3c", "weak": "#3498db",
               "compensatory": "#f39c12", "neurogenic": "#9b59b6"}
    sns.boxplot(data=df, x="condition", y="geodesic_length", order=order,
                palette=palette, ax=ax)
    sns.stripplot(data=df, x="condition", y="geodesic_length", order=order,
                  color="black", alpha=0.3, size=3, ax=ax)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Geodesic Length L(γ)")
    ax.set_title("Geodesic Length by Dysphagia Phenotype")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp03_geodesic_length.png", dpi=cfg["output"]["dpi"])
    plt.close()

    # --- Table ---
    summary = df.groupby("condition")["geodesic_length"].agg(["mean", "std", "median"])
    summary.to_csv(f"{tab_dir}/exp03_geodesic_length.csv")
    print(f"  Mean geodesic length by condition:\n{summary}")


# =====================================================================
# Experiment 4: Curvature profiles
# =====================================================================
def exp04_curvature(cohort, cfg, fig_dir, tab_dir):
    """Validate curvature as coordination complexity measure."""
    print("\n[EXP04] Curvature profile analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    conditions = ["healthy", "fibrotic", "weak", "compensatory", "neurogenic"]
    colors = {"healthy": "#2ecc71", "fibrotic": "#e74c3c", "weak": "#3498db",
              "compensatory": "#f39c12", "neurogenic": "#9b59b6"}

    data = []
    for idx, cond in enumerate(conditions):
        ax = axes.flat[idx]
        cond_trajs = [t for t in cohort if t.condition == cond]
        for traj in cond_trajs[:10]:
            smoothed = traj.smooth()
            ax.plot(smoothed.time, smoothed.curvature,
                    color=colors[cond], alpha=0.3, linewidth=0.8)
        ax.set_title(f"{cond.capitalize()}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("κ(t)")
        ax.grid(True, alpha=0.3)

        for traj in cond_trajs:
            data.append({
                "condition": cond,
                "mean_curvature": mean_curvature(traj.smooth()),
                "total_curvature": traj.smooth().total_curvature(),
            })

    # Summary panel
    ax = axes.flat[5]
    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="condition", y="mean_curvature", order=conditions,
                palette=colors, ax=ax)
    ax.set_title("Mean Curvature by Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Mean κ")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp04_curvature.png", dpi=cfg["output"]["dpi"])
    plt.close()

    summary = df.groupby("condition")[["mean_curvature", "total_curvature"]].agg(["mean", "std"])
    summary.to_csv(f"{tab_dir}/exp04_curvature.csv")
    print(f"  Curvature summary:\n{summary}")


# =====================================================================
# Experiment 5: Phase region detection
# =====================================================================
def exp05_phase_regions(cohort, cfg, fig_dir, tab_dir):
    """Validate geometric phase detection accuracy."""
    print("\n[EXP05] Phase region detection...")

    healthy = [t for t in cohort if t.condition == "healthy"][:20]
    results = []
    for traj in healthy:
        smoothed = traj.smooth().interpolate(200)
        phases = detect_phases_geometric(smoothed)
        pm = phase_metrics(smoothed, phases)
        bn_score = bottleneck_traversal_score(smoothed, phases)
        results.append({
            "n_phases_detected": len(phases),
            "bottleneck_score": bn_score,
            **{f"{k}_duration": v["duration"] for k, v in pm.items()},
        })

    df = pd.DataFrame(results)
    df.to_csv(f"{tab_dir}/exp05_phase_detection.csv", index=False)

    # Figure: phase duration distributions
    duration_cols = [c for c in df.columns if c.endswith("_duration")]
    if duration_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[duration_cols].boxplot(ax=ax)
        ax.set_ylabel("Duration (s)")
        ax.set_title("Detected Phase Durations (Healthy Swallows)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/exp05_phase_regions.png", dpi=cfg["output"]["dpi"])
        plt.close()

    print(f"  Mean phases detected: {df['n_phases_detected'].mean():.1f}")
    print(f"  Mean bottleneck score: {df['bottleneck_score'].mean():.3f}")


# =====================================================================
# Experiment 6: SRVF time-warp invariance
# =====================================================================
def exp06_srvf_invariance(cohort, cfg, fig_dir, tab_dir):
    """Validate temporal reparameterization invariance of SRVF."""
    print("\n[EXP06] SRVF time-warp invariance test...")

    healthy = [t for t in cohort if t.condition == "healthy"][:5]
    results = []

    for traj in healthy:
        smoothed = traj.smooth()
        res = time_warp_invariance_test(
            smoothed,
            n_warpings=cfg["srvf"]["n_warpings"],
            warp_strength=cfg["srvf"]["warp_strength"],
        )
        results.append(res)

    # Aggregate
    all_srvf = np.concatenate([r["srvf_distances"] for r in results])
    all_eucl = np.concatenate([r["euclidean_distances"] for r in results])
    ratios = [r["invariance_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(all_srvf, bins=20, alpha=0.7, color="#3498db", label="SRVF dist")
    ax.hist(all_eucl, bins=20, alpha=0.7, color="#e74c3c", label="Euclidean dist")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distances to Time-Warped Versions")
    ax.legend()

    ax = axes[1]
    ax.bar(["SRVF", "Euclidean"],
           [np.mean(all_srvf), np.mean(all_eucl)],
           yerr=[np.std(all_srvf), np.std(all_eucl)],
           color=["#3498db", "#e74c3c"], capsize=5)
    ax.set_ylabel("Mean Distance")
    ax.set_title(f"Invariance Ratio: {np.mean(ratios):.1f}x")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp06_srvf_invariance.png", dpi=cfg["output"]["dpi"])
    plt.close()

    summary = {
        "mean_srvf_distance": float(np.mean(all_srvf)),
        "std_srvf_distance": float(np.std(all_srvf)),
        "mean_euclidean_distance": float(np.mean(all_eucl)),
        "std_euclidean_distance": float(np.std(all_eucl)),
        "invariance_ratio": float(np.mean(ratios)),
    }
    pd.DataFrame([summary]).to_csv(f"{tab_dir}/exp06_srvf_invariance.csv", index=False)
    print(f"  SRVF mean dist: {summary['mean_srvf_distance']:.4f}")
    print(f"  Euclidean mean dist: {summary['mean_euclidean_distance']:.4f}")
    print(f"  Invariance ratio: {summary['invariance_ratio']:.1f}x")


# =====================================================================
# Experiment 7: Inter-structure synchrony
# =====================================================================
def exp07_synchrony(cohort, cfg, fig_dir, tab_dir):
    """Validate inter-structure synchrony metrics."""
    print("\n[EXP07] Inter-structure synchrony...")

    data = []
    for traj in cohort:
        metrics = extract_all_metrics(traj.smooth(), DEFAULT_LANDMARK_GROUPS)
        metrics["condition"] = traj.condition
        data.append(metrics)

    df = pd.DataFrame(data)

    # Plot synchrony-related metrics
    sync_cols = [c for c in df.columns if "coupling" in c or "phase_lag" in c or "synchrony" in c or "energy" in c]
    if not sync_cols:
        sync_cols = ["synchrony_risk", "energy_sharing"]
    sync_cols = [c for c in sync_cols if c in df.columns]

    if sync_cols:
        fig, axes = plt.subplots(1, len(sync_cols), figsize=(6 * len(sync_cols), 5))
        if len(sync_cols) == 1:
            axes = [axes]
        conditions = ["healthy", "fibrotic", "weak", "compensatory", "neurogenic"]
        palette = {"healthy": "#2ecc71", "fibrotic": "#e74c3c", "weak": "#3498db",
                   "compensatory": "#f39c12", "neurogenic": "#9b59b6"}

        for ax, col in zip(axes, sync_cols):
            if col in df.columns:
                sns.boxplot(data=df, x="condition", y=col, order=conditions,
                            palette=palette, ax=ax)
                ax.set_title(col.replace("_", " ").title())
                ax.tick_params(axis="x", rotation=30)
                ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{fig_dir}/exp07_synchrony.png", dpi=cfg["output"]["dpi"])
        plt.close()

    df.to_csv(f"{tab_dir}/exp07_synchrony.csv", index=False)
    print("  Synchrony metrics computed and saved.")


# =====================================================================
# Experiment 8: Phenotype discrimination
# =====================================================================
def exp08_phenotypes(cohort, cfg, fig_dir, tab_dir):
    """Validate phenotype classification from manifold metrics."""
    print("\n[EXP08] Phenotype discrimination...")

    classifier = PhenotypeClassifier(method=cfg["classification"]["method"])
    df = classifier.extract_features(cohort, DEFAULT_LANDMARK_GROUPS)

    feature_cols = classifier.feature_names_
    X = df[feature_cols].values
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    label_map = {"healthy": 0, "fibrotic": 1, "weak": 2,
                 "compensatory": 3, "neurogenic": 4}
    y = df["condition"].map(label_map).values

    # Cross-validation
    cv_results = classifier.cross_validate(X, y, n_folds=cfg["classification"]["n_cv_folds"])
    print(f"  CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")

    # Full fit for confusion matrix
    classifier.fit(X, y)
    eval_results = classifier.evaluate(X, y)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = eval_results["confusion_matrix"]
    conditions = ["Healthy", "Fibrotic", "Weak", "Compensatory", "Neurogenic"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=conditions, yticklabels=conditions, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Phenotype Classification (CV Acc: {cv_results['mean_accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp08_phenotypes.png", dpi=cfg["output"]["dpi"])
    plt.close()

    # Feature importance
    if classifier.feature_importances_ is not None:
        imp = pd.Series(classifier.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        imp.plot(kind="barh", ax=ax, color="#3498db")
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top 15 Manifold Metric Features for Phenotype Classification")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/exp08_feature_importance.png", dpi=cfg["output"]["dpi"])
        plt.close()

    # Save results
    pd.DataFrame([{
        "cv_accuracy_mean": cv_results["mean_accuracy"],
        "cv_accuracy_std": cv_results["std_accuracy"],
        "macro_auc": eval_results["macro_auc"],
    }]).to_csv(f"{tab_dir}/exp08_phenotype_results.csv", index=False)

    report_df = pd.DataFrame(eval_results["classification_report"]).T
    report_df.to_csv(f"{tab_dir}/exp08_classification_report.csv")


# =====================================================================
# Comprehensive metric summary table
# =====================================================================
def generate_summary_table(cohort, cfg, tab_dir):
    """Generate Table 1: Manifold metric summary by group."""
    print("\n[SUMMARY] Generating comprehensive metric table...")

    data = []
    for traj in cohort:
        metrics = extract_all_metrics(traj.smooth(), DEFAULT_LANDMARK_GROUPS)
        metrics["condition"] = traj.condition
        data.append(metrics)

    df = pd.DataFrame(data)
    numeric_cols = [c for c in df.columns if c != "condition"]

    summary = df.groupby("condition")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(f"{tab_dir}/table1_metric_summary.csv")
    print("  Saved comprehensive metric summary table.")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Run all simulated experiments")
    parser.add_argument("--config", default="configs/simulation_config.yaml")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments to run (e.g., 01 03 08)")
    args = parser.parse_args()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    fig_dir = cfg["output"]["figures_dir"]
    tab_dir = cfg["output"]["tables_dir"]

    print("=" * 70)
    print("SWALLOWING MANIFOLD ANALYSIS — SIMULATED EXPERIMENTS")
    print("=" * 70)

    # Generate synthetic data
    print("\nGenerating synthetic cohort...")
    t0 = time.time()
    synth_manifold = SyntheticManifold(
        intrinsic_dim=cfg["manifold"]["intrinsic_dim"],
        ambient_dim=cfg["manifold"]["ambient_dim"],
        seed=cfg["manifold"]["seed"],
    )
    generator = TrajectoryGenerator(
        manifold=synth_manifold,
        n_frames=cfg["trajectory"]["n_frames"],
        fps=cfg["trajectory"]["fps"],
        seed=cfg["manifold"]["seed"],
    )
    cohort = generator.generate_cohort(
        n_per_condition=cfg["cohort"]["n_per_condition"],
        conditions=cfg["cohort"]["conditions"],
        noise_std=cfg["trajectory"]["noise_std"],
    )
    print(f"  Generated {len(cohort)} trajectories in {time.time()-t0:.1f}s")

    # Run experiments
    experiments = {
        "01": ("Manifold dimensionality", lambda: exp01_manifold_dim(cohort, cfg, fig_dir, tab_dir)),
        "02": ("Trajectory types", lambda: exp02_trajectory_types(cohort, cfg, fig_dir, tab_dir)),
        "03": ("Geodesic length", lambda: exp03_geodesic_length(cohort, cfg, fig_dir, tab_dir)),
        "04": ("Curvature profiles", lambda: exp04_curvature(cohort, cfg, fig_dir, tab_dir)),
        "05": ("Phase regions", lambda: exp05_phase_regions(cohort, cfg, fig_dir, tab_dir)),
        "06": ("SRVF invariance", lambda: exp06_srvf_invariance(cohort, cfg, fig_dir, tab_dir)),
        "07": ("Synchrony", lambda: exp07_synchrony(cohort, cfg, fig_dir, tab_dir)),
        "08": ("Phenotypes", lambda: exp08_phenotypes(cohort, cfg, fig_dir, tab_dir)),
    }

    to_run = args.experiments or list(experiments.keys())
    for exp_id in sorted(to_run):
        if exp_id in experiments:
            name, fn = experiments[exp_id]
            t0 = time.time()
            try:
                fn()
                print(f"  ✓ Completed in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    generate_summary_table(cohort, cfg, tab_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures: {fig_dir}/")
    print(f"Tables:  {tab_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
