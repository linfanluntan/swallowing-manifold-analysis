"""
Synthetic swallowing manifold and trajectory generators.

Generates ground-truth trajectories for healthy and pathological
swallowing, embedding them in a high-dimensional landmark space
with known manifold dimension, curvature, and phase structure.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.trajectory import SwallowingTrajectory


# Default landmark groups (14 landmarks x 2D = 28D ambient space)
DEFAULT_LANDMARK_GROUPS = {
    "tongue": list(range(0, 8)),      # 4 landmarks x 2 coords
    "larynx": list(range(8, 16)),     # 4 landmarks x 2 coords
    "pharynx": list(range(16, 24)),   # 4 landmarks x 2 coords
    "ues": list(range(24, 28)),       # 2 landmarks x 2 coords
}
N_LANDMARKS = 14
AMBIENT_DIM = N_LANDMARKS * 2  # 2D coordinates


class SyntheticManifold:
    """
    Generate a synthetic swallowing manifold.

    The manifold is constructed as a smooth submanifold of R^{28}
    with known intrinsic dimension d=5, capturing the five coupled
    degrees of freedom (tongue, hyoid, larynx, pharynx, UES).
    """

    def __init__(
        self,
        intrinsic_dim: int = 5,
        ambient_dim: int = AMBIENT_DIM,
        seed: int = 42,
    ):
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.rng = np.random.RandomState(seed)

        # Random smooth embedding: intrinsic -> ambient
        # Using random orthogonal mixing + nonlinear terms
        self.A = self._random_orthogonal_matrix()
        self.b = self.rng.randn(ambient_dim) * 0.1

    def _random_orthogonal_matrix(self) -> np.ndarray:
        """Generate a random matrix for linear embedding component."""
        M = self.rng.randn(self.ambient_dim, self.intrinsic_dim)
        Q, _ = np.linalg.qr(M)
        return Q[:, : self.intrinsic_dim]

    def embed(self, z: np.ndarray) -> np.ndarray:
        """
        Map from intrinsic coordinates z to ambient landmark space.

        Includes nonlinear terms to ensure manifold curvature.
        """
        z = np.atleast_2d(z)
        squeeze = (z.shape[0] == 1)

        # Linear component
        x = z @ self.A.T + self.b

        # Nonlinear coupling terms (curvature sources)
        for i in range(min(self.intrinsic_dim, 3)):
            for j in range(i + 1, min(self.intrinsic_dim, 4)):
                coupling = 0.1 * np.sin(z[:, i] * z[:, j])
                idx_a = 2 * i % self.ambient_dim
                idx_b = (2 * j + 1) % self.ambient_dim
                x[:, idx_a] += coupling
                x[:, idx_b] += coupling

        if squeeze:
            x = x.squeeze(axis=0)
        return x


class TrajectoryGenerator:
    """
    Generate synthetic swallowing trajectories on the manifold.

    Creates healthy and pathological trajectories with known
    properties for validation experiments.
    """

    def __init__(
        self,
        manifold: SyntheticManifold,
        n_frames: int = 100,
        fps: float = 25.0,
        seed: int = 42,
    ):
        self.manifold = manifold
        self.n_frames = n_frames
        self.fps = fps
        self.rng = np.random.RandomState(seed)

    def _intrinsic_trajectory(
        self,
        condition: str = "healthy",
        params: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Generate a trajectory in intrinsic coordinates (d-dimensional).

        The trajectory passes through 5 phase regions with
        characteristic dynamics.
        """
        d = self.manifold.intrinsic_dim
        T = self.n_frames
        t = np.linspace(0, 1, T)

        z = np.zeros((T, d))

        if condition == "healthy":
            # Smooth, coordinated, canonical path
            z[:, 0] = np.sin(np.pi * t) * 2.0           # Tongue propulsion
            z[:, 1] = np.sin(np.pi * (t - 0.1)) * 1.8   # Laryngeal elevation
            z[:, 2] = np.sin(np.pi * (t - 0.15)) * 1.5  # Pharyngeal constriction
            z[:, 3] = np.sin(np.pi * (t - 0.2)) * 1.2   # Epiglottic inversion
            if d > 4:
                z[:, 4] = np.sin(np.pi * (t - 0.25)) * 1.0  # UES opening

        elif condition == "fibrotic":
            # Contracted manifold: reduced range, low curvature
            scale = params.get("severity", 0.5) if params else 0.5
            z[:, 0] = np.sin(np.pi * t) * 2.0 * scale
            z[:, 1] = np.sin(np.pi * (t - 0.1)) * 1.8 * scale
            z[:, 2] = np.sin(np.pi * (t - 0.15)) * 1.5 * scale
            z[:, 3] = np.sin(np.pi * (t - 0.2)) * 1.2 * scale
            if d > 4:
                z[:, 4] = np.sin(np.pi * (t - 0.25)) * 1.0 * scale

        elif condition == "weak":
            # Preserved shape but low speed (slow traversal)
            slow = params.get("slowdown", 0.6) if params else 0.6
            t_slow = t ** (1.0 / slow)  # Stretch early phase
            z[:, 0] = np.sin(np.pi * t_slow) * 2.0
            z[:, 1] = np.sin(np.pi * (t_slow - 0.1)) * 1.8
            z[:, 2] = np.sin(np.pi * (t_slow - 0.15)) * 1.5
            z[:, 3] = np.sin(np.pi * (t_slow - 0.2)) * 1.2
            if d > 4:
                z[:, 4] = np.sin(np.pi * (t_slow - 0.25)) * 1.0

        elif condition == "compensatory":
            # Alternate path: higher effort, curvature spikes
            z[:, 0] = np.sin(np.pi * t) * 2.5           # Excessive tongue
            z[:, 1] = np.sin(np.pi * (t - 0.2)) * 1.4   # Delayed larynx
            z[:, 2] = np.sin(np.pi * (t - 0.1)) * 2.0   # Extra pharyngeal
            z[:, 3] = np.sin(np.pi * (t - 0.3)) * 0.8   # Reduced epiglottic
            if d > 4:
                z[:, 4] = np.sin(np.pi * (t - 0.15)) * 1.5
            # Add compensatory head motion (curvature spike)
            z[:, 0] += 0.5 * np.exp(-((t - 0.4) ** 2) / 0.01)

        elif condition == "neurogenic":
            # Unstable control: base trajectory + random perturbations
            trial_noise = params.get("noise_level", 0.4) if params else 0.4
            z[:, 0] = np.sin(np.pi * t) * 2.0
            z[:, 1] = np.sin(np.pi * (t - 0.1)) * 1.8
            z[:, 2] = np.sin(np.pi * (t - 0.15)) * 1.5
            z[:, 3] = np.sin(np.pi * (t - 0.2)) * 1.2
            if d > 4:
                z[:, 4] = np.sin(np.pi * (t - 0.25)) * 1.0
            # Add structured noise (poor motor control)
            for dim in range(d):
                z[:, dim] += trial_noise * self.rng.randn(T)
                # Smooth the noise slightly
                from scipy.ndimage import gaussian_filter1d
                z[:, dim] = gaussian_filter1d(z[:, dim], sigma=2)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        return z

    def generate(
        self,
        condition: str = "healthy",
        params: Optional[Dict] = None,
        noise_std: float = 0.02,
        subject_id: str = "",
    ) -> SwallowingTrajectory:
        """
        Generate a single synthetic swallowing trajectory.

        Parameters
        ----------
        condition : str
            'healthy', 'fibrotic', 'weak', 'compensatory', 'neurogenic'
        params : dict, optional
            Condition-specific parameters.
        noise_std : float
            Measurement noise level.
        subject_id : str

        Returns
        -------
        SwallowingTrajectory
        """
        # Generate in intrinsic coordinates
        z = self._intrinsic_trajectory(condition, params)

        # Embed in ambient landmark space
        landmarks = np.array([self.manifold.embed(z[t]) for t in range(self.n_frames)])

        # Add measurement noise
        landmarks += self.rng.randn(*landmarks.shape) * noise_std

        time = np.arange(self.n_frames) / self.fps

        return SwallowingTrajectory(
            landmarks=landmarks,
            time=time,
            fps=self.fps,
            subject_id=subject_id or f"{condition}_{self.rng.randint(10000)}",
            condition=condition,
        )

    def generate_cohort(
        self,
        n_per_condition: int = 50,
        conditions: Optional[List[str]] = None,
        noise_std: float = 0.02,
    ) -> List[SwallowingTrajectory]:
        """
        Generate a cohort of trajectories across conditions.

        Returns list of SwallowingTrajectory objects.
        """
        if conditions is None:
            conditions = ["healthy", "fibrotic", "weak", "compensatory", "neurogenic"]

        trajectories = []
        for cond in conditions:
            for i in range(n_per_condition):
                # Vary parameters slightly for each subject
                params = {}
                if cond == "fibrotic":
                    params["severity"] = 0.3 + 0.4 * self.rng.rand()
                elif cond == "weak":
                    params["slowdown"] = 0.4 + 0.4 * self.rng.rand()
                elif cond == "neurogenic":
                    params["noise_level"] = 0.2 + 0.4 * self.rng.rand()

                traj = self.generate(
                    condition=cond,
                    params=params,
                    noise_std=noise_std,
                    subject_id=f"{cond}_{i:03d}",
                )
                trajectories.append(traj)

        return trajectories
