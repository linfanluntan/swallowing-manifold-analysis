"""
Manifold-based functional metrics for swallowing trajectories.

Implements the metrics from Sections 3 and 5 of the paper:
- Geodesic length (effort/efficiency)
- Curvature profiles (coordination complexity)
- Inter-structure synchrony (coupled manifold dynamics)
- Bottleneck traversal (safety)
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional

from .trajectory import SwallowingTrajectory


def geodesic_length(traj: SwallowingTrajectory) -> float:
    """
    Compute the geodesic length L(gamma) = integral ||gamma_dot|| dt.

    This measures total distance in physiological state space,
    approximating neuromuscular effort.
    """
    return traj.arc_length()


def mean_curvature(traj: SwallowingTrajectory) -> float:
    """Mean geodesic curvature along the trajectory."""
    return float(np.mean(traj.curvature))


def curvature_profile(traj: SwallowingTrajectory) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time, curvature) profile for the trajectory."""
    return traj.time, traj.curvature


def velocity_profile(traj: SwallowingTrajectory) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time, speed) profile for the trajectory."""
    return traj.time, traj.speed


def peak_velocity(traj: SwallowingTrajectory) -> float:
    """Maximum tangent magnitude (peak motion speed)."""
    return float(np.max(traj.speed))


def time_to_peak_velocity(traj: SwallowingTrajectory) -> float:
    """Time of maximum speed (seconds from swallow onset)."""
    idx = np.argmax(traj.speed)
    return float(traj.time[idx] - traj.time[0])


# =====================================================================
# Inter-structure synchrony metrics (Section 5.3)
# =====================================================================

def activation_envelope(traj: SwallowingTrajectory) -> np.ndarray:
    """
    Compute activation envelope a(t) = ||gamma_dot(t)||
    for a subsystem trajectory.
    """
    return traj.speed


def phase_lag(
    traj_a: SwallowingTrajectory,
    traj_b: SwallowingTrajectory,
    method: str = "peak",
) -> float:
    """
    Compute phase lag between two subsystem trajectories.

    Parameters
    ----------
    traj_a, traj_b : SwallowingTrajectory
        Subsystem trajectories (e.g., tongue, larynx).
    method : str
        'peak' — difference in peak activation times.
        'xcorr' — lag at maximum cross-correlation.

    Returns
    -------
    lag : float
        Phase lag in seconds (positive = a leads b).
    """
    env_a = activation_envelope(traj_a)
    env_b = activation_envelope(traj_b)

    if method == "peak":
        t_peak_a = traj_a.time[np.argmax(env_a)]
        t_peak_b = traj_b.time[np.argmax(env_b)]
        return float(t_peak_b - t_peak_a)

    elif method == "xcorr":
        # Normalize envelopes
        a_norm = (env_a - np.mean(env_a)) / (np.std(env_a) + 1e-12)
        b_norm = (env_b - np.mean(env_b)) / (np.std(env_b) + 1e-12)
        xcorr = correlate(a_norm, b_norm, mode="full")
        lags = np.arange(-len(a_norm) + 1, len(a_norm))
        dt = np.mean(np.diff(traj_a.time))
        best_lag = lags[np.argmax(xcorr)] * dt
        return float(best_lag)

    else:
        raise ValueError(f"Unknown method: {method}")


def coupling_strength(
    traj_a: SwallowingTrajectory,
    traj_b: SwallowingTrajectory,
) -> float:
    """
    Tangent-space correlation between two subsystem trajectories.

    C_ab = corr(||v_a(t)||, ||v_b(t)||)

    Values near 1 indicate tight coordination; near 0 indicates
    independence; negative values indicate antagonistic timing.
    """
    env_a = activation_envelope(traj_a)
    env_b = activation_envelope(traj_b)

    # Ensure same length
    n = min(len(env_a), len(env_b))
    r, _ = pearsonr(env_a[:n], env_b[:n])
    return float(r)


def energy_sharing_index(
    subsystem_trajs: Dict[str, SwallowingTrajectory],
) -> float:
    """
    Energy sharing index: measures how much subsystems co-activate.

    E_share = integral(sum_{k!=j} ||v_k|| ||v_j||) /
              integral((sum_k ||v_k||)^2)

    High values indicate overlapping, cooperative activation.
    """
    envelopes = [activation_envelope(t) for t in subsystem_trajs.values()]
    n = min(len(e) for e in envelopes)
    envelopes = [e[:n] for e in envelopes]

    K = len(envelopes)
    cross_term = np.zeros(n)
    for i in range(K):
        for j in range(K):
            if i != j:
                cross_term += envelopes[i] * envelopes[j]

    total_term = np.sum(envelopes, axis=0) ** 2

    return float(np.sum(cross_term) / (np.sum(total_term) + 1e-12))


def synchrony_risk_functional(
    subsystem_trajs: Dict[str, SwallowingTrajectory],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Synchrony risk functional R_sync (Section 5.3.8).

    R_sync = w1 * |lag_T->L| + w2 * (1 - C_TL) + w3 * (1 - C_LP)

    Higher values indicate greater coordination failure risk.
    """
    if weights is None:
        weights = {"lag": 1.0, "coupling_tl": 1.0, "coupling_lp": 1.0}

    names = list(subsystem_trajs.keys())
    if len(names) < 2:
        return 0.0

    # Default subsystem mapping
    tongue = subsystem_trajs.get("tongue", list(subsystem_trajs.values())[0])
    larynx = subsystem_trajs.get("larynx", list(subsystem_trajs.values())[1])
    pharynx = subsystem_trajs.get(
        "pharynx", list(subsystem_trajs.values())[-1]
    )

    lag_tl = abs(phase_lag(tongue, larynx))
    c_tl = coupling_strength(tongue, larynx)
    c_lp = coupling_strength(larynx, pharynx)

    risk = (
        weights["lag"] * lag_tl
        + weights["coupling_tl"] * (1.0 - c_tl)
        + weights["coupling_lp"] * (1.0 - c_lp)
    )
    return float(risk)


# =====================================================================
# Comprehensive metric extraction
# =====================================================================

def extract_all_metrics(
    traj: SwallowingTrajectory,
    landmark_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Extract all manifold-based metrics from a swallowing trajectory.

    Returns a dictionary of named metric values.
    """
    metrics = {
        "geodesic_length": geodesic_length(traj),
        "mean_curvature": mean_curvature(traj),
        "total_curvature": traj.total_curvature(),
        "peak_velocity": peak_velocity(traj),
        "time_to_peak_velocity": time_to_peak_velocity(traj),
        "smoothness_index": traj.smoothness_index(),
        "duration": float(traj.time[-1] - traj.time[0]),
    }

    # Subsystem metrics if landmark groups provided
    if landmark_groups is not None and len(landmark_groups) >= 2:
        subs = traj.subsystem_trajectories(landmark_groups)
        names = list(subs.keys())

        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                lag = phase_lag(subs[n1], subs[n2])
                coup = coupling_strength(subs[n1], subs[n2])
                metrics[f"phase_lag_{n1}_{n2}"] = lag
                metrics[f"coupling_{n1}_{n2}"] = coup

        metrics["energy_sharing"] = energy_sharing_index(subs)
        metrics["synchrony_risk"] = synchrony_risk_functional(subs)

    return metrics
