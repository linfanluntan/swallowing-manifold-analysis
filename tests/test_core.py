"""
Unit tests for core manifold, trajectory, and metrics modules.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.trajectory import SwallowingTrajectory
from core.metrics import (
    geodesic_length, mean_curvature, phase_lag,
    coupling_strength, extract_all_metrics,
)
from core.manifold import SwallowingManifold
from core.srvf import compute_srvf, srvf_distance
from core.phase_detection import detect_phases_geometric


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def simple_trajectory():
    """A simple sinusoidal trajectory for testing."""
    T, D = 100, 10
    t = np.linspace(0, 1, T)
    landmarks = np.column_stack([
        np.sin(2 * np.pi * t * (k + 1)) for k in range(D)
    ])
    return SwallowingTrajectory(landmarks, t, fps=100.0, subject_id="test", condition="healthy")


@pytest.fixture
def straight_trajectory():
    """A straight-line trajectory (zero curvature)."""
    T, D = 100, 10
    t = np.linspace(0, 1, T)
    landmarks = np.column_stack([t * (k + 1) for k in range(D)])
    return SwallowingTrajectory(landmarks, t, fps=100.0)


@pytest.fixture
def two_subsystem_trajs():
    """Two trajectories representing coupled subsystems."""
    T = 100
    t = np.linspace(0, 1, T)
    # Tongue: peaks at t=0.3
    tongue_lm = np.column_stack([np.sin(np.pi * t) * 2, np.cos(np.pi * t)])
    # Larynx: peaks at t=0.4 (slight lag)
    larynx_lm = np.column_stack([np.sin(np.pi * (t - 0.1)) * 1.5, np.cos(np.pi * (t - 0.1))])

    tongue = SwallowingTrajectory(tongue_lm, t, fps=100.0)
    larynx = SwallowingTrajectory(larynx_lm, t, fps=100.0)
    return tongue, larynx


# =====================================================================
# Trajectory tests
# =====================================================================

class TestTrajectory:
    def test_creation(self, simple_trajectory):
        assert simple_trajectory.n_frames == 100
        assert simple_trajectory.n_dims == 10

    def test_velocity_shape(self, simple_trajectory):
        v = simple_trajectory.velocity
        assert v.shape == (100, 10)

    def test_speed_nonnegative(self, simple_trajectory):
        assert np.all(simple_trajectory.speed >= 0)

    def test_curvature_nonnegative(self, simple_trajectory):
        smoothed = simple_trajectory.smooth()
        assert np.all(smoothed.curvature >= 0)

    def test_arc_length_positive(self, simple_trajectory):
        L = simple_trajectory.arc_length()
        assert L > 0

    def test_straight_line_low_curvature(self, straight_trajectory):
        """Straight trajectory should have near-zero curvature."""
        kappa = straight_trajectory.curvature
        assert np.mean(kappa) < 0.1

    def test_smooth_preserves_shape(self, simple_trajectory):
        smoothed = simple_trajectory.smooth()
        assert smoothed.n_frames == simple_trajectory.n_frames
        assert smoothed.n_dims == simple_trajectory.n_dims

    def test_interpolate(self, simple_trajectory):
        interp = simple_trajectory.interpolate(200)
        assert interp.n_frames == 200
        assert interp.n_dims == simple_trajectory.n_dims

    def test_subsystem_decomposition(self, simple_trajectory):
        groups = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}
        subs = simple_trajectory.subsystem_trajectories(groups)
        assert len(subs) == 2
        assert subs["a"].n_dims == 5
        assert subs["b"].n_dims == 5


# =====================================================================
# Metrics tests
# =====================================================================

class TestMetrics:
    def test_geodesic_length(self, simple_trajectory):
        L = geodesic_length(simple_trajectory)
        assert isinstance(L, float)
        assert L > 0

    def test_mean_curvature(self, simple_trajectory):
        kappa = mean_curvature(simple_trajectory.smooth())
        assert isinstance(kappa, float)
        assert kappa >= 0

    def test_phase_lag(self, two_subsystem_trajs):
        tongue, larynx = two_subsystem_trajs
        lag = phase_lag(tongue, larynx, method="peak")
        # Larynx should lag tongue by ~0.1 seconds
        assert isinstance(lag, float)

    def test_coupling_strength_range(self, two_subsystem_trajs):
        tongue, larynx = two_subsystem_trajs
        c = coupling_strength(tongue, larynx)
        assert -1 <= c <= 1

    def test_extract_all_metrics(self, simple_trajectory):
        metrics = extract_all_metrics(simple_trajectory)
        assert "geodesic_length" in metrics
        assert "mean_curvature" in metrics
        assert "peak_velocity" in metrics


# =====================================================================
# Manifold tests
# =====================================================================

class TestManifold:
    def test_pca_embedding(self):
        X = np.random.randn(200, 20)
        manifold = SwallowingManifold(n_components=5, method="pca")
        manifold.fit(X)
        assert manifold.embedding_.shape == (200, 5)

    def test_dimension_estimation(self):
        # Low-dimensional data in high-D space
        t = np.linspace(0, 2 * np.pi, 200)
        Z = np.column_stack([np.sin(t), np.cos(t), t])
        # Embed in 20D
        A = np.random.randn(20, 3)
        X = Z @ A.T + np.random.randn(200, 20) * 0.01

        manifold = SwallowingManifold(n_neighbors=10)
        result = manifold.estimate_intrinsic_dimension(X, max_dim=10, method="eigenvalue_gap")
        assert result["estimated_dim"] <= 5  # Should detect low dimensionality


# =====================================================================
# SRVF tests
# =====================================================================

class TestSRVF:
    def test_srvf_shape(self, simple_trajectory):
        q = compute_srvf(simple_trajectory.landmarks, simple_trajectory.time)
        assert q.shape == simple_trajectory.landmarks.shape

    def test_srvf_distance_nonnegative(self, simple_trajectory):
        d = srvf_distance(simple_trajectory, simple_trajectory)
        assert d >= 0

    def test_srvf_self_distance_small(self, simple_trajectory):
        d = srvf_distance(simple_trajectory, simple_trajectory)
        assert d < 0.1  # Self-distance should be near zero


# =====================================================================
# Phase detection tests
# =====================================================================

class TestPhaseDetection:
    def test_detects_phases(self, simple_trajectory):
        smoothed = simple_trajectory.smooth().interpolate(200)
        phases = detect_phases_geometric(smoothed)
        assert len(phases) >= 2  # Should detect at least 2 phases
        assert len(phases) <= 6  # Not more than 6

    def test_phases_cover_timeline(self, simple_trajectory):
        smoothed = simple_trajectory.smooth().interpolate(200)
        phases = detect_phases_geometric(smoothed)
        # Check that phases span the trajectory
        starts = [v[0] for v in phases.values()]
        ends = [v[1] for v in phases.values()]
        assert min(starts) <= smoothed.time[1]
        assert max(ends) >= smoothed.time[-2]
