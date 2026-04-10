"""
Geometric phase detection for swallowing trajectories.

Implements Section 4: phases as regions of the manifold rather than
temporal labels. Detects oral propulsion, pharyngeal contraction,
laryngeal closure, UES opening, and esophageal entry as geometric
regions characterized by curvature, velocity, and tangent direction.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional

from .trajectory import SwallowingTrajectory


# Phase labels
PHASE_NAMES = [
    "oral_propulsion",
    "pharyngeal_contraction",
    "laryngeal_closure",
    "ues_opening",
    "esophageal_entry",
]


def detect_phases_geometric(
    traj: SwallowingTrajectory,
    n_phases: int = 5,
    curvature_threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Detect swallowing phases from trajectory geometry.

    Uses curvature peaks and velocity profile to segment the
    trajectory into geometric regions.

    Parameters
    ----------
    traj : SwallowingTrajectory
    n_phases : int
        Expected number of phases (default: 5).
    curvature_threshold : float
        Relative threshold for curvature peak detection.

    Returns
    -------
    phases : dict
        Maps phase name to (start_time, end_time) tuple.
    """
    kappa = traj.curvature
    speed = traj.speed
    time = traj.time

    # Normalize curvature
    kappa_norm = kappa / (np.max(kappa) + 1e-12)

    # Find curvature peaks (phase transition points)
    min_distance = max(3, len(kappa) // (2 * n_phases))
    peaks, properties = find_peaks(
        kappa_norm,
        height=curvature_threshold * np.mean(kappa_norm),
        distance=min_distance,
    )

    # If not enough peaks, use velocity-based segmentation
    if len(peaks) < n_phases - 1:
        # Segment by velocity profile changes
        speed_norm = speed / (np.max(speed) + 1e-12)
        peaks_v, _ = find_peaks(speed_norm, distance=min_distance)
        troughs_v, _ = find_peaks(-speed_norm, distance=min_distance)
        transition_points = sorted(
            list(peaks_v) + list(troughs_v)
        )
        # Take top n_phases-1 most prominent
        if len(transition_points) >= n_phases - 1:
            peaks = np.array(transition_points[: n_phases - 1])
        else:
            # Fallback: uniform segmentation
            peaks = np.linspace(
                0, len(time) - 1, n_phases + 1
            ).astype(int)[1:-1]

    # Sort and limit to n_phases - 1 transition points
    peaks = np.sort(peaks)[: n_phases - 1]

    # Build phase intervals
    boundaries = [0] + list(peaks) + [len(time) - 1]
    phases = {}
    for i, name in enumerate(PHASE_NAMES[:n_phases]):
        if i < len(boundaries) - 1:
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            phases[name] = (float(time[start_idx]), float(time[end_idx]))

    return phases


def phase_labels_from_regions(
    traj: SwallowingTrajectory,
    phases: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Assign a phase label to each time frame.

    Returns
    -------
    labels : ndarray of shape (T,), dtype=int
        Phase index (0 to n_phases-1) for each frame.
    """
    labels = np.zeros(traj.n_frames, dtype=int)
    for i, (name, (t_start, t_end)) in enumerate(phases.items()):
        mask = (traj.time >= t_start) & (traj.time <= t_end)
        labels[mask] = i
    return labels


def phase_metrics(
    traj: SwallowingTrajectory,
    phases: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-phase metrics.

    Returns
    -------
    dict mapping phase name to metric dict with:
        'duration', 'mean_speed', 'mean_curvature', 'path_length'
    """
    results = {}
    for name, (t_start, t_end) in phases.items():
        mask = (traj.time >= t_start) & (traj.time <= t_end)
        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        sub_speed = traj.speed[idx]
        sub_curv = traj.curvature[idx]
        sub_time = traj.time[idx]

        # Path length in this phase
        dt = np.diff(sub_time)
        avg_speed = 0.5 * (sub_speed[:-1] + sub_speed[1:])
        path_len = float(np.sum(avg_speed * dt)) if len(dt) > 0 else 0.0

        results[name] = {
            "duration": float(t_end - t_start),
            "mean_speed": float(np.mean(sub_speed)),
            "mean_curvature": float(np.mean(sub_curv)),
            "path_length": path_len,
        }

    return results


def bottleneck_traversal_score(
    traj: SwallowingTrajectory,
    phases: Dict[str, Tuple[float, float]],
    bottleneck_phase: str = "laryngeal_closure",
) -> float:
    """
    Score how well the trajectory traverses the laryngeal closure
    bottleneck (Section 4.5).

    Higher scores indicate more robust bottleneck traversal.
    Score combines: sufficient speed through the region,
    adequate dwell time, and smooth passage.
    """
    if bottleneck_phase not in phases:
        return 0.0

    t_start, t_end = phases[bottleneck_phase]
    mask = (traj.time >= t_start) & (traj.time <= t_end)

    if not np.any(mask):
        return 0.0

    idx = np.where(mask)[0]
    speed_in_bn = traj.speed[idx]
    curv_in_bn = traj.curvature[idx]

    # Components:
    # 1. Adequate speed (not stalling)
    speed_score = np.mean(speed_in_bn) / (np.max(traj.speed) + 1e-12)

    # 2. Smooth passage (low curvature variance)
    curv_var = np.var(curv_in_bn)
    smoothness_score = 1.0 / (1.0 + curv_var)

    # 3. Sufficient dwell time
    duration = t_end - t_start
    total_duration = traj.time[-1] - traj.time[0]
    dwell_score = min(1.0, duration / (0.15 * total_duration + 1e-12))

    return float(0.4 * speed_score + 0.3 * smoothness_score + 0.3 * dwell_score)
