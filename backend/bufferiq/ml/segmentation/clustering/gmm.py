"""Gaussian Mixture Model clustering."""

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture
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


class GMMClusterer:
    """Gaussian Mixture Model clustering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize GMM clusterer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.n_init = self.config.get("n_init", 10)
        self.random_state = self.config.get("random_state", 42)

    def fit(
        self, feature_matrix: np.ndarray, n_clusters: int, platform: str
    ) -> ClusteringResult:
        """
        Fit GMM clustering.

        Args:
            feature_matrix: Feature matrix
            n_clusters: Number of components
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
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            labels = gmm.fit_predict(feature_matrix)

            sil_score = silhouette_score(feature_matrix, labels)
            ch_score = calinski_harabasz_score(feature_matrix, labels)
            db_score = davies_bouldin_score(feature_matrix, labels)
            bic = gmm.bic(feature_matrix)

            # Estimate cluster centers
            cluster_centers = gmm.means_

            return ClusteringResult(
                algorithm="gmm",
                n_clusters=n_clusters,
                labels=labels,
                cluster_centers=cluster_centers,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=float(bic),
                convergence_iterations=gmm.n_iter_,
                noise_ratio=0.0,
                stability_score=sil_score,
                platform=platform,
            )
        except Exception as e:
            raise ClusteringError(f"GMM clustering failed: {str(e)}")