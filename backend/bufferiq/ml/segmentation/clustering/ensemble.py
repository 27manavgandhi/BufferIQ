"""Ensemble clustering combining multiple algorithms."""

from typing import Any, Dict, List

import numpy as np

from bufferiq.ml.segmentation.types import ClusteringResult, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ClusteringError,
    UnsupportedPlatformError,
)
from bufferiq.ml.segmentation.clustering.kmeans import KMeansClusterer
from bufferiq.ml.segmentation.clustering.hierarchical import HierarchicalClusterer
from bufferiq.ml.segmentation.clustering.gmm import GMMClusterer


class ClusteringEnsemble:
    """Ensemble clustering combining multiple algorithms."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize ensemble."""
        self.config = config or {}
        self.kmeans = KMeansClusterer(self.config.get("kmeans", {}))
        self.hierarchical = HierarchicalClusterer(self.config.get("hierarchical", {}))
        self.gmm = GMMClusterer(self.config.get("gmm", {}))

    def fit(
        self, feature_matrix: np.ndarray, n_clusters: int, platform: str
    ) -> ClusteringResult:
        """
        Fit ensemble clustering.

        Args:
            feature_matrix: Feature matrix
            n_clusters: Number of clusters
            platform: Platform type

        Returns:
            Best clustering result

        Raises:
            UnsupportedPlatformError: If platform not supported
            ClusteringError: If all clustering fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        results: List[ClusteringResult] = []

        # Try each algorithm
        try:
            results.append(self.kmeans.fit(feature_matrix, n_clusters, platform))
        except Exception:
            pass

        try:
            results.append(
                self.hierarchical.fit(feature_matrix, n_clusters, platform)
            )
        except Exception:
            pass

        try:
            results.append(self.gmm.fit(feature_matrix, n_clusters, platform))
        except Exception:
            pass

        if not results:
            raise ClusteringError("All clustering algorithms failed")

        # Return best result (highest silhouette score)
        return max(results, key=lambda r: r.silhouette_score)