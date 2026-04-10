"""
Trajectory representation and analysis on the swallowing manifold.

A swallow is modeled as a parametrized curve gamma: [0,T] -> M,
where M is the physiological configuration manifold. MRI frames
are discrete samples along this curve.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from typing import Optional, Tuple, List


class SwallowingTrajectory:
    """
    Represents a single swallowing trajectory as a curve in
    landmark space or manifold embedding space.

    Parameters
    ----------
    landmarks : ndarray of shape (T, D)
        Landmark positions over T time frames in D dimensions.
    time : ndarray of shape (T,), optional
        Physical time stamps. If None, uses frame indices.
    fps : float
        Frames per second of the acquisition.
    subject_id : str
        Subject identifier.
    condition : str
        Clinical condition label (e.g., 'healthy', 'fibrotic').
    """

    def __init__(
        self,
        landmarks: np.ndarray,
        time: Optional[np.ndarray] = None,
        fps: float = 25.0,
        subject_id: str = "",
        condition: str = "unknown",
    ):
        self.landmarks = np.asarray(landmarks, dtype=np.float64)
        self.n_frames, self.n_dims = self.landmarks.shape
        self.fps = fps
        self.subject_id = subject_id
        self.condition = condition

        if time is not None:
            self.time = np.asarray(time, dtype=np.float64)
        else:
            self.time = np.arange(self.n_frames) / self.fps

        # Computed on demand
        self._velocity = None
        self._acceleration = None
        self._speed = None
        self._curvature = None

    def smooth(self, window_length: int = 7, polyorder: int = 3) -> "SwallowingTrajectory":
        """Apply Savitzky-Golay smoothing to landmark trajectories."""
        if self.n_frames < window_length:
            return self
        smoothed = np.zeros_like(self.landmarks)
        for j in range(self.n_dims):
            smoothed[:, j] = savgol_filter(
                self.landmarks[:, j], window_length, polyorder
            )
        return SwallowingTrajectory(
            smoothed, self.time, self.fps, self.subject_id, self.condition
        )

    def interpolate(self, n_points: int = 200) -> "SwallowingTrajectory":
        """Resample trajectory to uniform time spacing via cubic spline."""
        t_new = np.linspace(self.time[0], self.time[-1], n_points)
        new_landmarks = np.zeros((n_points, self.n_dims))
        for j in range(self.n_dims):
            cs = CubicSpline(self.time, self.landmarks[:, j])
            new_landmarks[:, j] = cs(t_new)
        return SwallowingTrajectory(
            new_landmarks, t_new, self.fps, self.subject_id, self.condition
        )

    @property
    def velocity(self) -> np.ndarray:
        """Tangent vectors gamma_dot(t) via central differences."""
        if self._velocity is None:
            dt = np.diff(self.time)
            # Central differences (forward/backward at boundaries)
            self._velocity = np.zeros_like(self.landmarks)
            self._velocity[1:-1] = (
                self.landmarks[2:] - self.landmarks[:-2]
            ) / (self.time[2:] - self.time[:-2])[:, None]
            self._velocity[0] = (
                self.landmarks[1] - self.landmarks[0]
            ) / dt[0]
            self._velocity[-1] = (
                self.landmarks[-1] - self.landmarks[-2]
            ) / dt[-1]
        return self._velocity

    @property
    def speed(self) -> np.ndarray:
        """Tangent magnitude ||gamma_dot(t)|| at each time point."""
        if self._speed is None:
            self._speed = np.linalg.norm(self.velocity, axis=1)
        return self._speed

    @property
    def acceleration(self) -> np.ndarray:
        """Second derivative (covariant acceleration approximation)."""
        if self._acceleration is None:
            dt = np.diff(self.time)
            self._acceleration = np.zeros_like(self.landmarks)
            self._acceleration[1:-1] = (
                self.velocity[2:] - self.velocity[:-2]
            ) / (self.time[2:] - self.time[:-2])[:, None]
            if len(dt) > 0:
                self._acceleration[0] = (
                    self.velocity[1] - self.velocity[0]
                ) / dt[0]
                self._acceleration[-1] = (
                    self.velocity[-1] - self.velocity[-2]
                ) / dt[-1]
        return self._acceleration

    @property
    def curvature(self) -> np.ndarray:
        """
        Geodesic curvature kappa(t) = ||acceleration|| / ||velocity||^2.

        This is the Euclidean approximation; for manifold curvature,
        project acceleration onto the tangent space.
        """
        if self._curvature is None:
            acc_norm = np.linalg.norm(self.acceleration, axis=1)
            speed_sq = self.speed ** 2
            # Avoid division by zero
            safe_speed = np.maximum(speed_sq, 1e-12)
            self._curvature = acc_norm / safe_speed
        return self._curvature

    def arc_length(self) -> float:
        """
        Total geodesic length L(gamma) = integral of ||gamma_dot|| dt.

        This is the primary measure of swallowing effort/efficiency.
        """
        dt = np.diff(self.time)
        avg_speed = 0.5 * (self.speed[:-1] + self.speed[1:])
        return float(np.sum(avg_speed * dt))

    def total_curvature(self) -> float:
        """Integrated curvature: integral of kappa(t) dt."""
        dt = np.diff(self.time)
        avg_curv = 0.5 * (self.curvature[:-1] + self.curvature[1:])
        return float(np.sum(avg_curv * dt))

    def smoothness_index(self) -> float:
        """
        Smoothness as negative mean log dimensionless jerk.

        Higher values indicate smoother trajectories.
        """
        jerk = np.diff(self.acceleration, axis=0)
        # dt for jerk: between acceleration time points (skip first and last)
        dt = np.diff(self.time)
        dt_jerk = dt[1:]  # length = T-2, matches jerk length T-2
        if len(dt_jerk) < len(jerk):
            dt_jerk = np.append(dt_jerk, dt_jerk[-1] if len(dt_jerk) > 0 else 1.0)
        dt_jerk = dt_jerk[:len(jerk)]
        jerk_rate = jerk / np.maximum(dt_jerk[:, None], 1e-12)
        jerk_norm = np.linalg.norm(jerk_rate, axis=1)
        # Normalize by mean speed
        mean_speed = np.mean(self.speed) + 1e-12
        duration = self.time[-1] - self.time[0] + 1e-12
        njerk = np.sqrt(
            duration ** 3 * np.mean(jerk_norm ** 2)
        ) / mean_speed
        return -np.log(njerk + 1e-12)

    def subsystem_trajectories(
        self, landmark_groups: dict
    ) -> dict:
        """
        Decompose into subsystem trajectories.

        Parameters
        ----------
        landmark_groups : dict
            Maps subsystem name to list of column indices.
            Example: {'tongue': [0,1,2,3], 'larynx': [4,5,6,7]}

        Returns
        -------
        dict mapping subsystem name to SwallowingTrajectory
        """
        result = {}
        for name, indices in landmark_groups.items():
            sub_landmarks = self.landmarks[:, indices]
            result[name] = SwallowingTrajectory(
                sub_landmarks, self.time, self.fps,
                self.subject_id, self.condition
            )
        return result
