"""DBSCAN clustering implementation."""

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

from bufferiq.ml.segmentation.types import ClusteringResult, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ClusteringError,
    UnsupportedPlatformError,
)


class DBSCANClusterer:
    """Density-based DBSCAN clustering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize DBSCAN clusterer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.eps = self.config.get("eps", 0.5)
        self.min_samples = self.config.get("min_samples", 5)

    def fit(
        self, feature_matrix: np.ndarray, platform: str
    ) -> ClusteringResult:
        """
        Fit DBSCAN clustering.

        Args:
            feature_matrix: Feature matrix
            platform: Platform type

        Returns:
            Clustering result

        Raises:
            UnsupportedPlatformError: If platform not supported
            ClusteringError: If clustering fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        try:
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(feature_matrix)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float(np.sum(labels == -1)) / len(labels)

            if n_clusters > 1:
                # Filter out noise points for metrics calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 0:
                    non_noise_labels = labels[non_noise_mask]
                    non_noise_data = feature_matrix[non_noise_mask]
                    sil_score = silhouette_score(non_noise_data, non_noise_labels)
                    db_score = davies_bouldin_score(non_noise_data, non_noise_labels)
                else:
                    sil_score = 0.0
                    db_score = 0.0
            else:
                sil_score = 0.0
                db_score = 0.0

            return ClusteringResult(
                algorithm="dbscan",
                n_clusters=max(n_clusters, 1),
                labels=labels,
                cluster_centers=None,
                silhouette_score=sil_score,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=db_score,
                inertia=None,
                convergence_iterations=None,
                noise_ratio=noise_ratio,
                stability_score=1.0 - noise_ratio,
                platform=platform,
            )
        except Exception as e:
            raise ClusteringError(f"DBSCAN clustering failed: {str(e)}")