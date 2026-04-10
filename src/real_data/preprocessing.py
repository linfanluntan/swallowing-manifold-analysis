"""
Preprocessing pipeline for real cine MRI swallowing data.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from typing import List, Dict, Optional

from ..core.trajectory import SwallowingTrajectory


def smooth_landmarks(
    landmarks: np.ndarray,
    window_length: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to each landmark coordinate."""
    smoothed = np.zeros_like(landmarks)
    for j in range(landmarks.shape[1]):
        if landmarks.shape[0] >= window_length:
            smoothed[:, j] = savgol_filter(landmarks[:, j], window_length, polyorder)
        else:
            smoothed[:, j] = landmarks[:, j]
    return smoothed


def normalize_centroid(landmarks: np.ndarray) -> np.ndarray:
    """Center landmarks by subtracting the centroid at each frame."""
    n_frames, n_dims = landmarks.shape
    n_landmarks = n_dims // 2
    for t in range(n_frames):
        xs = landmarks[t, 0::2]
        ys = landmarks[t, 1::2]
        cx, cy = np.mean(xs), np.mean(ys)
        landmarks[t, 0::2] -= cx
        landmarks[t, 1::2] -= cy
    return landmarks


def remove_outlier_frames(
    landmarks: np.ndarray,
    time: np.ndarray,
    threshold: float = 3.0,
) -> tuple:
    """Remove frames with landmark jumps exceeding threshold * IQR."""
    diffs = np.sqrt(np.sum(np.diff(landmarks, axis=0)**2, axis=1))
    q1, q3 = np.percentile(diffs, [25, 75])
    iqr = q3 - q1
    cutoff = q3 + threshold * iqr
    mask = np.ones(len(landmarks), dtype=bool)
    outlier_frames = np.where(diffs > cutoff)[0] + 1
    mask[outlier_frames] = False
    return landmarks[mask], time[mask]


def preprocess_trajectory(
    traj: SwallowingTrajectory,
    smooth_window: int = 7,
    smooth_order: int = 3,
    normalize: bool = True,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    n_interpolate: int = 200,
) -> SwallowingTrajectory:
    """Full preprocessing pipeline for a single trajectory."""
    landmarks = traj.landmarks.copy()
    time = traj.time.copy()

    if remove_outliers and len(landmarks) > 10:
        landmarks, time = remove_outlier_frames(landmarks, time, outlier_threshold)

    if normalize:
        landmarks = normalize_centroid(landmarks)

    landmarks = smooth_landmarks(landmarks, smooth_window, smooth_order)

    # Interpolate to uniform time
    t_new = np.linspace(time[0], time[-1], n_interpolate)
    new_lm = np.zeros((n_interpolate, landmarks.shape[1]))
    for j in range(landmarks.shape[1]):
        cs = CubicSpline(time, landmarks[:, j])
        new_lm[:, j] = cs(t_new)

    return SwallowingTrajectory(
        new_lm, t_new, traj.fps, traj.subject_id, traj.condition
    )
