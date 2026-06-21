"""K-Means clustering implementation."""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

from bufferiq.ml.segmentation.types import ClusteringResult, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ClusteringError,
    UnsupportedPlatformError,
)


class KMeansClusterer:
    """K-Means clustering with stability testing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize K-Means clusterer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.n_init = self.config.get("n_init", 10)
        self.max_iter = self.config.get("max_iter", 300)
        self.random_state = self.config.get("random_state", 42)

    def fit(
        self, feature_matrix: np.ndarray, n_clusters: int, platform: str
    ) -> ClusteringResult:
        """
        Fit K-Means clustering.

        Args:
            feature_matrix: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            platform: Platform type

        Returns:
            Clustering result with metrics

        Raises:
            UnsupportedPlatformError: If platform not supported
            ClusteringError: If clustering fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if feature_matrix.shape[0] < n_clusters:
            raise ClusteringError(
                f"Number of samples ({feature_matrix.shape[0]}) "
                f"must be >= n_clusters ({n_clusters})"
            )

        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = kmeans.fit_predict(feature_matrix)

            sil_score = silhouette_score(feature_matrix, labels)
            ch_score = calinski_harabasz_score(feature_matrix, labels)
            db_score = davies_bouldin_score(feature_matrix, labels)
            stability = self._compute_stability(feature_matrix, n_clusters)

            return ClusteringResult(
                algorithm="kmeans",
                n_clusters=n_clusters,
                labels=labels,
                cluster_centers=kmeans.cluster_centers_,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=kmeans.inertia_,
                convergence_iterations=kmeans.n_iter_,
                noise_ratio=0.0,
                stability_score=stability,
                platform=platform,
            )
        except Exception as e:
            raise ClusteringError(f"K-Means clustering failed: {str(e)}")

    def _compute_stability(
        self, feature_matrix: np.ndarray, n_clusters: int, n_runs: int = 5
    ) -> float:
        """Compute clustering stability via multiple runs."""
        base_labels = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=self.n_init
        ).fit_predict(feature_matrix)

        stability_scores = []
        for seed in range(1, n_runs + 1):
            labels = KMeans(
                n_clusters=n_clusters, random_state=seed * 10, n_init=self.n_init
            ).fit_predict(feature_matrix)
            stability_scores.append(adjusted_rand_score(base_labels, labels))

        return float(np.mean(stability_scores))