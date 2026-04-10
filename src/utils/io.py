"""Utility functions for configuration and I/O."""

import yaml
import os
import numpy as np
import pandas as pd
from typing import Dict, List


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_trajectories_csv(trajectories, filepath: str):
    """Save a list of trajectories to CSV."""
    rows = []
    for traj in trajectories:
        for i in range(traj.n_frames):
            row = {"subject_id": traj.subject_id, "condition": traj.condition,
                   "frame": i, "time": traj.time[i]}
            for j in range(traj.n_dims):
                row[f"x{j+1}"] = traj.landmarks[i, j]
            rows.append(row)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def load_trajectories_csv(filepath: str, fps: float = 25.0):
    """Load trajectories from CSV."""
    from ..core.trajectory import SwallowingTrajectory

    df = pd.read_csv(filepath)
    trajectories = []
    for subj, sdf in df.groupby("subject_id"):
        sdf = sdf.sort_values("frame")
        coord_cols = [c for c in sdf.columns if c.startswith("x")]
        landmarks = sdf[coord_cols].values.astype(np.float64)
        time = sdf["time"].values if "time" in sdf.columns else sdf["frame"].values / fps
        condition = sdf["condition"].iloc[0] if "condition" in sdf.columns else "unknown"
        trajectories.append(SwallowingTrajectory(
            landmarks, time, fps, str(subj), condition
        ))
    return trajectories
