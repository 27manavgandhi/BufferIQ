"""Hierarchical clustering implementation."""

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from bufferiq.ml.segmentation.types import ClusteringResult, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ClusteringError,
    UnsupportedPlatformError,
)


class HierarchicalClusterer:
    """Hierarchical clustering with Ward linkage."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize hierarchical clusterer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.linkage = self.config.get("linkage", "ward")

    def fit(
        self, feature_matrix: np.ndarray, n_clusters: int, platform: str
    ) -> ClusteringResult:
        """
        Fit hierarchical clustering.

        Args:
            feature_matrix: Feature matrix
            n_clusters: Number of clusters
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
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=self.linkage
            )
            labels = clusterer.fit_predict(feature_matrix)

            sil_score = silhouette_score(feature_matrix, labels)
            ch_score = calinski_harabasz_score(feature_matrix, labels)
            db_score = davies_bouldin_score(feature_matrix, labels)

            return ClusteringResult(
                algorithm="hierarchical",
                n_clusters=n_clusters,
                labels=labels,
                cluster_centers=None,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=None,
                convergence_iterations=None,
                noise_ratio=0.0,
                stability_score=sil_score,
                platform=platform,
            )
        except Exception as e:
            raise ClusteringError(f"Hierarchical clustering failed: {str(e)}")