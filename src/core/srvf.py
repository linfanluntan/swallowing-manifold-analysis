"""
Square Root Velocity Function (SRVF) for elastic trajectory comparison.

Implements temporally reparameterization-invariant distances between
swallowing trajectories (Section 5.2 of the paper).

Based on: Srivastava et al., IEEE TPAMI 2011;
          Su et al., Ann. Appl. Stat. 2014.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional

from .trajectory import SwallowingTrajectory


def compute_srvf(gamma: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Compute the Square Root Velocity Function of a curve.

    q(t) = gamma_dot(t) / sqrt(||gamma_dot(t)||)

    Parameters
    ----------
    gamma : ndarray of shape (T, D)
        Curve in D-dimensional space sampled at T points.
    time : ndarray of shape (T,)
        Time parameterization.

    Returns
    -------
    q : ndarray of shape (T, D)
        SRVF representation.
    """
    T, D = gamma.shape
    dt = np.diff(time)
    # Velocity via finite differences
    v = np.diff(gamma, axis=0) / dt[:, None]

    # SRVF: q = v / sqrt(||v||)
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    safe_speed = np.maximum(speed, 1e-12)
    q = v / np.sqrt(safe_speed)

    # Pad to same length (replicate last value)
    q = np.vstack([q, q[-1:]])
    return q


def srvf_distance(
    traj1: SwallowingTrajectory,
    traj2: SwallowingTrajectory,
    n_resample: int = 200,
) -> float:
    """
    Compute the elastic (SRVF) distance between two trajectories.

    This distance is invariant to temporal reparameterization,
    comparing trajectory shape rather than speed.

    Parameters
    ----------
    traj1, traj2 : SwallowingTrajectory
    n_resample : int
        Number of points for uniform resampling.

    Returns
    -------
    dist : float
        L2 distance in SRVF space.
    """
    # Resample to uniform parameterization
    t1 = traj1.interpolate(n_resample)
    t2 = traj2.interpolate(n_resample)

    q1 = compute_srvf(t1.landmarks, t1.time)
    q2 = compute_srvf(t2.landmarks, t2.time)

    # L2 distance in SRVF space
    diff = q1 - q2
    dist = np.sqrt(np.sum(diff ** 2) / n_resample)
    return float(dist)


def srvf_distance_with_alignment(
    traj1: SwallowingTrajectory,
    traj2: SwallowingTrajectory,
    n_resample: int = 200,
    n_dp_grid: int = 100,
) -> Tuple[float, np.ndarray]:
    """
    Compute elastic distance with optimal time warping via
    dynamic programming.

    d(gamma1, gamma2) = inf_phi ||q1 - (q2 o phi) * sqrt(phi_dot)||

    Parameters
    ----------
    traj1, traj2 : SwallowingTrajectory
    n_resample : int
    n_dp_grid : int
        Grid resolution for DP alignment.

    Returns
    -------
    dist : float
        Aligned SRVF distance.
    warping : ndarray
        Optimal warping function phi.
    """
    t1 = traj1.interpolate(n_resample)
    t2 = traj2.interpolate(n_resample)

    q1 = compute_srvf(t1.landmarks, t1.time)
    q2 = compute_srvf(t2.landmarks, t2.time)

    # Dynamic programming alignment
    N = n_dp_grid
    T = n_resample

    # Resample SRVFs to DP grid
    idx1 = np.linspace(0, T - 1, N).astype(int)
    q1_grid = q1[idx1]
    q2_grid = q2[idx1]

    # DP cost matrix
    D_mat = np.full((N, N), np.inf)
    D_mat[0, 0] = np.sum((q1_grid[0] - q2_grid[0]) ** 2)

    for i in range(1, N):
        for j in range(1, N):
            cost = np.sum((q1_grid[i] - q2_grid[j]) ** 2)
            D_mat[i, j] = cost + min(
                D_mat[i - 1, j - 1],
                D_mat[i - 1, j],
                D_mat[i, j - 1],
            )

    # Backtrack to find optimal warping
    warping = np.zeros(N, dtype=int)
    i, j = N - 1, N - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        candidates = []
        if i > 0 and j > 0:
            candidates.append((i - 1, j - 1, D_mat[i - 1, j - 1]))
        if i > 0:
            candidates.append((i - 1, j, D_mat[i - 1, j]))
        if j > 0:
            candidates.append((i, j - 1, D_mat[i, j - 1]))

        best = min(candidates, key=lambda x: x[2])
        i, j = best[0], best[1]
        path.append((i, j))

    path.reverse()
    warping = np.array([p[1] for p in path])

    # Compute aligned distance
    aligned_dist = np.sqrt(D_mat[N - 1, N - 1] / N)

    return float(aligned_dist), warping


def srvf_distance_matrix(
    trajectories: list,
    aligned: bool = True,
    n_resample: int = 200,
) -> np.ndarray:
    """
    Compute pairwise SRVF distance matrix for a set of trajectories.

    Parameters
    ----------
    trajectories : list of SwallowingTrajectory
    aligned : bool
        If True, use DP-aligned elastic distance.
    n_resample : int

    Returns
    -------
    D : ndarray of shape (N, N)
        Pairwise distance matrix.
    """
    N = len(trajectories)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            if aligned:
                d, _ = srvf_distance_with_alignment(
                    trajectories[i], trajectories[j], n_resample
                )
            else:
                d = srvf_distance(
                    trajectories[i], trajectories[j], n_resample
                )
            D[i, j] = d
            D[j, i] = d

    return D


def time_warp_invariance_test(
    traj: SwallowingTrajectory,
    n_warpings: int = 10,
    warp_strength: float = 0.3,
    n_resample: int = 200,
) -> dict:
    """
    Validate that SRVF distance is invariant to time reparameterization.

    Creates n_warpings random temporal warpings of the trajectory and
    verifies that SRVF distance to the original is near zero, while
    Euclidean distance is large.

    Returns
    -------
    dict with:
        'srvf_distances': array of SRVF distances to warped versions
        'euclidean_distances': array of L2 distances
        'invariance_ratio': mean(euclidean) / mean(srvf)
    """
    srvf_dists = []
    eucl_dists = []

    t_norm = np.linspace(0, 1, traj.n_frames)

    for _ in range(n_warpings):
        # Generate random monotonic warping
        phi = _random_warping(traj.n_frames, warp_strength)
        warped_time = traj.time[0] + phi * (traj.time[-1] - traj.time[0])

        # Interpolate trajectory at warped times
        warped_landmarks = np.zeros_like(traj.landmarks)
        for j in range(traj.n_dims):
            cs = CubicSpline(traj.time, traj.landmarks[:, j])
            warped_landmarks[:, j] = cs(warped_time)

        warped_traj = SwallowingTrajectory(
            warped_landmarks, warped_time, traj.fps,
            traj.subject_id, traj.condition
        )

        # SRVF distance (should be near zero for pure reparameterization)
        sd, _ = srvf_distance_with_alignment(traj, warped_traj, n_resample)
        srvf_dists.append(sd)

        # Euclidean L2 distance (will be large due to temporal misalignment)
        ed = np.sqrt(np.mean((traj.landmarks - warped_landmarks) ** 2))
        eucl_dists.append(ed)

    srvf_dists = np.array(srvf_dists)
    eucl_dists = np.array(eucl_dists)

    return {
        "srvf_distances": srvf_dists,
        "euclidean_distances": eucl_dists,
        "invariance_ratio": float(
            np.mean(eucl_dists) / (np.mean(srvf_dists) + 1e-12)
        ),
    }


def _random_warping(n: int, strength: float = 0.3) -> np.ndarray:
    """Generate a random monotonic warping function on [0, 1]."""
    t = np.linspace(0, 1, n)
    # Random perturbation
    noise = np.cumsum(np.random.exponential(1.0, n))
    noise = noise / noise[-1]  # Normalize to [0, 1]
    # Blend with identity
    phi = (1 - strength) * t + strength * noise
    phi = (phi - phi[0]) / (phi[-1] - phi[0])  # Ensure [0, 1]
    return phi
