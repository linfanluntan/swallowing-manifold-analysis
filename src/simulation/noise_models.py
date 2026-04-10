"""
Measurement noise and physiological variability models.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_noise(shape, std=0.02, rng=None):
    """Additive Gaussian measurement noise."""
    rng = rng or np.random.RandomState()
    return rng.randn(*shape) * std


def correlated_noise(n_frames, n_dims, std=0.02, correlation_length=5, rng=None):
    """Temporally correlated noise (smooth noise)."""
    rng = rng or np.random.RandomState()
    raw = rng.randn(n_frames, n_dims) * std
    smoothed = np.zeros_like(raw)
    for j in range(n_dims):
        smoothed[:, j] = gaussian_filter1d(raw[:, j], sigma=correlation_length)
    # Rescale to desired std
    current_std = np.std(smoothed)
    if current_std > 0:
        smoothed *= std / current_std
    return smoothed


def physiological_variability(n_frames, n_dims, scale=0.1, rng=None):
    """Inter-trial physiological variability (slow drift + jitter)."""
    rng = rng or np.random.RandomState()
    # Slow drift
    t = np.linspace(0, 1, n_frames)
    drift = np.outer(np.sin(2 * np.pi * t * rng.uniform(0.5, 2.0)), rng.randn(n_dims))
    drift *= scale
    # Fast jitter
    jitter = rng.randn(n_frames, n_dims) * scale * 0.3
    return drift + jitter
