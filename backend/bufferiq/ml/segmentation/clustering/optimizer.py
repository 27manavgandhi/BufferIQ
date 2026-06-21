"""Clustering optimization and configuration."""

from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from bufferiq.ml.segmentation.types import OptimalClusterConfig, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ClusteringError,
    UnsupportedPlatformError,
)


class ClusteringOptimizer:
    """
    Find optimal number of clusters and algorithm.

    Uses multiple validation metrics:
    - Silhouette score
    - Calinski-Harabasz index
    - Davies-Bouldin index
    - BIC (for GMM)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize optimizer."""
        self.config = config or {}
        self.min_clusters = self.config.get("min_clusters", 2)
        self.max_clusters = self.config.get("max_clusters", 10)

    def find_optimal(
        self, feature_matrix: np.ndarray, platform: str
    ) -> OptimalClusterConfig:
        """
        Find optimal clustering configuration.

        Args:
            feature_matrix: Feature matrix
            platform: Platform type

        Returns:
            Optimal cluster configuration

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if feature_matrix.shape[0] < self.min_clusters:
            raise ClusteringError(
                f"Need at least {self.min_clusters} samples for clustering"
            )

        scores: Dict[str, Dict[int, float]] = {
            "silhouette": {},
            "calinski_harabasz": {},
            "davies_bouldin": {},
            "bic": {},
            "inertia": {},
        }

        k_range = range(
            self.min_clusters, min(self.max_clusters + 1, feature_matrix.shape[0])
        )

        for k in k_range:
            # K-Means evaluation
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_matrix)

            scores["silhouette"][k] = silhouette_score(feature_matrix, labels)
            scores["calinski_harabasz"][k] = calinski_harabasz_score(
                feature_matrix, labels
            )
            scores["davies_bouldin"][k] = davies_bouldin_score(feature_matrix, labels)
            scores["inertia"][k] = kmeans.inertia_

            # GMM BIC
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(feature_matrix)
            scores["bic"][k] = gmm.bic(feature_matrix)

        optimal_k = self._find_optimal_k(scores, k_range)

        return OptimalClusterConfig(
            algorithm="kmeans",
            n_clusters=optimal_k,
            parameters={"n_init": 10, "max_iter": 300},
            validation_scores={
                k: v[optimal_k]
                for k, v in scores.items()
                if optimal_k in v
            },
            confidence=scores["silhouette"].get(optimal_k, 0.0),
        )

    def _find_optimal_k(
        self,
        scores: Dict[str, Dict[int, float]],
        k_range: range,
    ) -> int:
        """Find optimal k using composite scoring."""
        composite_scores: Dict[int, float] = {}

        for k in k_range:
            score = 0.0

            # Silhouette (higher is better, weight: 0.4)
            sil_values = list(scores["silhouette"].values())
            if sil_values:
                sil_norm = (scores["silhouette"].get(k, 0) - min(sil_values)) / (
                    max(sil_values) - min(sil_values) + 1e-10
                )
                score += sil_norm * 0.4

            # Calinski-Harabasz (higher is better, weight: 0.3)
            ch_values = list(scores["calinski_harabasz"].values())
            if ch_values:
                ch_norm = (scores["calinski_harabasz"].get(k, 0) - min(ch_values)) / (
                    max(ch_values) - min(ch_values) + 1e-10
                )
                score += ch_norm * 0.3

            # Davies-Bouldin (lower is better, weight: 0.3)
            db_values = list(scores["davies_bouldin"].values())
            if db_values:
                db_norm = 1.0 - (
                    scores["davies_bouldin"].get(k, 0) - min(db_values)
                ) / (max(db_values) - min(db_values) + 1e-10)
                score += db_norm * 0.3

            composite_scores[k] = score

        return max(composite_scores, key=composite_scores.__getitem__)