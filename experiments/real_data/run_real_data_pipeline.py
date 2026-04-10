#!/usr/bin/env python3
"""
Real data analysis pipeline.

End-to-end pipeline for analyzing cine MRI swallowing data:
1. Load and preprocess landmark time series
2. Learn manifold structure
3. Compute trajectory metrics
4. Compare with clinical scores
5. Longitudinal analysis (if data available)

Usage:
    python experiments/real_data/run_real_data_pipeline.py --config configs/real_data_config.yaml

Data Format:
    Place CSV files in data/real/raw/ with columns:
    subject_id, condition, frame, x1, y1, x2, y2, ..., x14, y14
    See docs/DATA_FORMAT.md for details.
"""

import argparse
import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.manifold import SwallowingManifold
from core.trajectory import SwallowingTrajectory
from core.metrics import extract_all_metrics
from core.srvf import srvf_distance_matrix
from core.phase_detection import detect_phases_geometric, phase_metrics, bottleneck_traversal_score
from core.phenotype import PhenotypeClassifier


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["tables_dir"], exist_ok=True)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)


# =====================================================================
# Step 1: Load and preprocess
# =====================================================================
def step01_load_and_preprocess(cfg):
    """Load landmark CSV files and preprocess."""
    print("\n[STEP 1] Loading and preprocessing data...")

    raw_dir = cfg["data"]["raw_dir"]
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))

    if not csv_files:
        print(f"  WARNING: No CSV files found in {raw_dir}")
        print("  Generating demo data for pipeline testing...")
        return _generate_demo_data(cfg)

    trajectories = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)

        # Expected columns: subject_id, condition, frame, x1, y1, ..., x14, y14
        subjects = df["subject_id"].unique()

        for subj in subjects:
            subj_df = df[df["subject_id"] == subj].sort_values("frame")
            coord_cols = [c for c in subj_df.columns
                          if c not in ("subject_id", "condition", "frame", "time")]

            landmarks = subj_df[coord_cols].values.astype(np.float64)
            condition = subj_df["condition"].iloc[0] if "condition" in subj_df.columns else "unknown"

            if "time" in subj_df.columns:
                time = subj_df["time"].values.astype(np.float64)
            else:
                time = subj_df["frame"].values / cfg["data"]["fps"]

            traj = SwallowingTrajectory(
                landmarks=landmarks,
                time=time,
                fps=cfg["data"]["fps"],
                subject_id=str(subj),
                condition=condition,
            )

            # Preprocessing
            prep = cfg["preprocessing"]
            traj = traj.smooth(
                window_length=prep["smoothing"]["window_length"],
                polyorder=prep["smoothing"]["polyorder"],
            )
            traj = traj.interpolate(prep["interpolation"]["n_points"])

            trajectories.append(traj)

    print(f"  Loaded {len(trajectories)} trajectories from {len(csv_files)} files")
    return trajectories


def _generate_demo_data(cfg):
    """Generate small demo dataset when no real data is available."""
    from simulation.trajectory_generator import SyntheticManifold, TrajectoryGenerator

    manifold = SyntheticManifold(intrinsic_dim=5, seed=123)
    gen = TrajectoryGenerator(manifold, n_frames=100, fps=25.0, seed=123)

    trajs = []
    for cond in ["healthy", "fibrotic"]:
        for i in range(10):
            t = gen.generate(condition=cond, subject_id=f"demo_{cond}_{i}")
            trajs.append(t)

    print(f"  Generated {len(trajs)} demo trajectories")
    return trajs


# =====================================================================
# Step 2: Manifold learning
# =====================================================================
def step02_manifold_learning(trajectories, cfg):
    """Learn manifold from trajectory data."""
    print("\n[STEP 2] Manifold learning...")

    X = np.vstack([t.landmarks for t in trajectories])
    ml_cfg = cfg["manifold_learning"]

    manifold = SwallowingManifold(
        n_components=5,
        method=ml_cfg["method"],
        n_neighbors=ml_cfg["n_neighbors"],
    )

    # Estimate intrinsic dimension
    dim_result = manifold.estimate_intrinsic_dimension(
        X, max_dim=ml_cfg["max_dim_test"]
    )
    print(f"  Estimated intrinsic dimension: {dim_result['estimated_dim']}")

    # Fit with estimated dimension
    manifold.n_components = min(dim_result["estimated_dim"], 5)
    manifold.fit(X)

    # Visualization
    fig_dir = cfg["output"]["figures_dir"]
    vis_manifold = SwallowingManifold(n_components=3, method="pca")
    vis_manifold.fit(X)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    conditions = list(set(t.condition for t in trajectories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    color_map = dict(zip(conditions, colors))

    for traj in trajectories[:30]:  # Plot subset
        embedded = vis_manifold.transform(traj.landmarks)
        ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                color=color_map[traj.condition], alpha=0.5, linewidth=1)

    for cond, color in color_map.items():
        ax.plot([], [], [], color=color, linewidth=2, label=cond)
    ax.legend()
    ax.set_title("Real Data: Trajectory Manifold Embedding")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/real_manifold_embedding.png", dpi=cfg["output"]["dpi"])
    plt.close()

    return manifold, dim_result


# =====================================================================
# Step 3: Trajectory analysis
# =====================================================================
def step03_trajectory_analysis(trajectories, cfg):
    """Compute all manifold metrics for each trajectory."""
    print("\n[STEP 3] Computing trajectory metrics...")

    landmark_groups = cfg.get("landmark_groups", None)

    all_metrics = []
    for traj in trajectories:
        metrics = extract_all_metrics(traj, landmark_groups)
        metrics["subject_id"] = traj.subject_id
        metrics["condition"] = traj.condition

        # Phase detection
        phases = detect_phases_geometric(traj)
        pm = phase_metrics(traj, phases)
        bn_score = bottleneck_traversal_score(traj, phases)
        metrics["bottleneck_score"] = bn_score
        metrics["n_phases"] = len(phases)

        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    tab_dir = cfg["output"]["tables_dir"]
    df.to_csv(f"{tab_dir}/real_trajectory_metrics.csv", index=False)

    # Summary statistics
    numeric_cols = [c for c in df.columns if c not in ("subject_id", "condition")]
    summary = df.groupby("condition")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(f"{tab_dir}/real_metric_summary.csv")
    print(f"  Computed metrics for {len(df)} trajectories")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")

    # Visualization: metric distributions
    fig_dir = cfg["output"]["figures_dir"]
    key_metrics = ["geodesic_length", "mean_curvature", "peak_velocity",
                   "smoothness_index", "bottleneck_score"]
    key_metrics = [m for m in key_metrics if m in df.columns]

    if len(key_metrics) > 0 and df["condition"].nunique() > 1:
        fig, axes = plt.subplots(1, len(key_metrics),
                                 figsize=(5 * len(key_metrics), 5))
        if len(key_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, key_metrics):
            sns.boxplot(data=df, x="condition", y=metric, ax=ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(f"{fig_dir}/real_metric_distributions.png",
                    dpi=cfg["output"]["dpi"])
        plt.close()

    return df


# =====================================================================
# Step 4: Clinical comparison
# =====================================================================
def step04_clinical_comparison(metrics_df, cfg):
    """Compare manifold metrics with clinical scores if available."""
    print("\n[STEP 4] Clinical score comparison...")

    scores_file = cfg.get("clinical_comparison", {}).get("scores_file", "")
    if not os.path.exists(scores_file):
        print(f"  Clinical scores file not found ({scores_file}).")
        print("  Generating simulated FOIS/MBSImP proxies for demonstration...")
        metrics_df = _simulate_clinical_scores(metrics_df)
    else:
        clinical_df = pd.read_csv(scores_file)
        metrics_df = metrics_df.merge(clinical_df, on="subject_id", how="left")

    # Correlate manifold metrics with clinical scores
    clinical_cols = [c for c in metrics_df.columns if c.startswith("FOIS") or c.startswith("MBSImP")]
    if not clinical_cols:
        clinical_cols = [c for c in metrics_df.columns if "clinical" in c.lower() or "fois" in c.lower()]

    metric_cols = cfg.get("clinical_comparison", {}).get("metrics_to_correlate", [])
    metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    if not metric_cols:
        metric_cols = ["geodesic_length", "mean_curvature", "peak_velocity"]
        metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    if clinical_cols and metric_cols:
        correlations = []
        for mc in metric_cols:
            for cc in clinical_cols:
                valid = metrics_df[[mc, cc]].dropna()
                if len(valid) > 5:
                    rho, pval = spearmanr(valid[mc], valid[cc])
                    correlations.append({
                        "manifold_metric": mc,
                        "clinical_score": cc,
                        "spearman_rho": rho,
                        "p_value": pval,
                    })

        corr_df = pd.DataFrame(correlations)
        tab_dir = cfg["output"]["tables_dir"]
        corr_df.to_csv(f"{tab_dir}/real_clinical_correlations.csv", index=False)
        print(f"  Computed {len(correlations)} correlations")
        if len(corr_df) > 0:
            print(corr_df.to_string(index=False))
    else:
        print("  No clinical comparison possible (missing columns).")


def _simulate_clinical_scores(df):
    """Generate proxy clinical scores from manifold metrics for demo."""
    if "geodesic_length" in df.columns:
        # FOIS proxy: inversely related to geodesic length
        gl = df["geodesic_length"].values
        gl_norm = (gl - gl.min()) / (gl.max() - gl.min() + 1e-12)
        df["FOIS_proxy"] = np.clip(7 - 6 * gl_norm + np.random.randn(len(df)) * 0.5, 1, 7).astype(int)

    if "mean_curvature" in df.columns:
        mc = df["mean_curvature"].values
        mc_norm = (mc - mc.min()) / (mc.max() - mc.min() + 1e-12)
        df["MBSImP_proxy"] = np.clip(mc_norm * 20 + np.random.randn(len(df)) * 2, 0, 30).astype(int)

    return df


# =====================================================================
# Step 5: Group comparison with statistics
# =====================================================================
def step05_group_comparison(metrics_df, cfg):
    """Statistical comparison between groups."""
    print("\n[STEP 5] Statistical group comparisons...")

    conditions = metrics_df["condition"].unique()
    if len(conditions) < 2:
        print("  Only one condition; skipping group comparison.")
        return

    key_metrics = ["geodesic_length", "mean_curvature", "peak_velocity",
                   "smoothness_index", "bottleneck_score"]
    key_metrics = [m for m in key_metrics if m in metrics_df.columns]

    results = []
    # Compare each condition pair
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            for metric in key_metrics:
                g1 = metrics_df[metrics_df["condition"] == c1][metric].dropna()
                g2 = metrics_df[metrics_df["condition"] == c2][metric].dropna()
                if len(g1) >= 3 and len(g2) >= 3:
                    stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                    # Effect size (rank-biserial)
                    n1, n2 = len(g1), len(g2)
                    effect_size = 1 - (2 * stat) / (n1 * n2)
                    results.append({
                        "group1": c1, "group2": c2, "metric": metric,
                        "U_statistic": stat, "p_value": pval,
                        "effect_size": effect_size,
                        "mean_g1": g1.mean(), "mean_g2": g2.mean(),
                    })

    results_df = pd.DataFrame(results)
    # Bonferroni correction
    if len(results_df) > 0:
        results_df["p_corrected"] = np.minimum(
            results_df["p_value"] * len(results_df), 1.0
        )
        results_df["significant"] = results_df["p_corrected"] < 0.05

    tab_dir = cfg["output"]["tables_dir"]
    results_df.to_csv(f"{tab_dir}/real_group_comparisons.csv", index=False)
    n_sig = results_df["significant"].sum() if "significant" in results_df.columns else 0
    print(f"  {n_sig}/{len(results_df)} comparisons significant after Bonferroni correction")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Run real data analysis pipeline")
    parser.add_argument("--config", default="configs/real_data_config.yaml")
    parser.add_argument("--steps", nargs="*", default=None,
                        help="Specific steps to run (1-5)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    print("=" * 70)
    print("SWALLOWING MANIFOLD ANALYSIS — REAL DATA PIPELINE")
    print("=" * 70)

    steps_to_run = [int(s) for s in args.steps] if args.steps else [1, 2, 3, 4, 5]

    trajectories = None
    metrics_df = None

    if 1 in steps_to_run:
        trajectories = step01_load_and_preprocess(cfg)

    if 2 in steps_to_run and trajectories:
        manifold, dim_result = step02_manifold_learning(trajectories, cfg)

    if 3 in steps_to_run and trajectories:
        metrics_df = step03_trajectory_analysis(trajectories, cfg)

    if 4 in steps_to_run and metrics_df is not None:
        step04_clinical_comparison(metrics_df, cfg)

    if 5 in steps_to_run and metrics_df is not None:
        step05_group_comparison(metrics_df, cfg)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Results: {cfg['output']['figures_dir']}/ and {cfg['output']['tables_dir']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
