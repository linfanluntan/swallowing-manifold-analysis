"""
Manifold learning and embedding for swallowing configuration spaces.

Implements the theoretical framework from Section 2 of the paper:
swallowing configurations lie on a low-dimensional manifold M embedded
in a high-dimensional ambient observation space.
"""

import numpy as np
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple, Dict


class SwallowingManifold:
    """
    Learn and represent the swallowing configuration manifold from
    landmark time-series data.

    Parameters
    ----------
    n_components : int
        Target intrinsic dimension d of the manifold.
    method : str
        Manifold learning method: 'isomap', 'diffusion_map', or 'pca'.
    n_neighbors : int
        Number of neighbors for graph-based methods.
    """

    def __init__(
        self,
        n_components: int = 5,
        method: str = "isomap",
        n_neighbors: int = 10,
    ):
        self.n_components = n_components
        self.method = method
        self.n_neighbors = n_neighbors
        self.embedding_ = None
        self.model_ = None
        self.ambient_dim_ = None
        self.reconstruction_error_ = None

    def fit(self, X: np.ndarray) -> "SwallowingManifold":
        """
        Learn the manifold embedding from data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            High-dimensional swallowing configurations. Each row is a
            flattened landmark vector at one time point.

        Returns
        -------
        self
        """
        self.ambient_dim_ = X.shape[1]

        if self.method == "isomap":
            self.model_ = Isomap(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
            )
            self.embedding_ = self.model_.fit_transform(X)
            self.reconstruction_error_ = self.model_.reconstruction_error()

        elif self.method == "diffusion_map":
            self.model_ = SpectralEmbedding(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                affinity="nearest_neighbors",
            )
            self.embedding_ = self.model_.fit_transform(X)
            self.reconstruction_error_ = None  # Not directly available

        elif self.method == "pca":
            self.model_ = PCA(n_components=self.n_components)
            self.embedding_ = self.model_.fit_transform(X)
            total_var = np.sum(self.model_.explained_variance_ratio_)
            self.reconstruction_error_ = 1.0 - total_var

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data onto the learned manifold."""
        if self.method == "pca":
            return self.model_.transform(X)
        elif self.method == "isomap":
            return self.model_.transform(X)
        else:
            raise NotImplementedError(
                f"Out-of-sample transform not available for {self.method}"
            )

    def estimate_intrinsic_dimension(
        self,
        X: np.ndarray,
        max_dim: int = 20,
        method: str = "residual_variance",
    ) -> Dict:
        """
        Estimate the intrinsic dimensionality of the data.

        Uses residual variance (Isomap) or PCA eigenvalue decay.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        max_dim : int
            Maximum dimension to test.
        method : str
            'residual_variance' or 'eigenvalue_gap'.

        Returns
        -------
        dict with keys:
            'dimensions': array of tested dimensions
            'scores': reconstruction quality at each dimension
            'estimated_dim': selected intrinsic dimension
        """
        dims = np.arange(1, min(max_dim + 1, X.shape[1]))
        scores = []

        if method == "residual_variance":
            # Compute geodesic distance matrix once
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(X)

            for d in dims:
                iso = Isomap(n_components=d, n_neighbors=self.n_neighbors)
                iso.fit(X)
                scores.append(1.0 - iso.reconstruction_error())

        elif method == "eigenvalue_gap":
            pca = PCA(n_components=min(max_dim, X.shape[1], X.shape[0]))
            pca.fit(X)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            scores = cumvar[: len(dims)]

        else:
            raise ValueError(f"Unknown method: {method}")

        scores = np.array(scores)

        # Estimate dimension via elbow detection
        if len(scores) > 2:
            # Second derivative approach
            d2 = np.diff(np.diff(scores))
            estimated_dim = int(np.argmax(np.abs(d2)) + 2)
        else:
            estimated_dim = 1

        return {
            "dimensions": dims[: len(scores)],
            "scores": scores,
            "estimated_dim": estimated_dim,
        }

    def geodesic_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise geodesic distances on the manifold.

        Uses graph shortest paths (Isomap-style) as an approximation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        D : ndarray of shape (n_samples, n_samples)
            Geodesic distance matrix.
        """
        if self.model_ is not None and hasattr(self.model_, "dist_matrix_"):
            return self.model_.dist_matrix_

        # Fallback: compute via Isomap internals
        iso = Isomap(n_components=2, n_neighbors=self.n_neighbors)
        iso.fit(X)
        return iso.dist_matrix_
