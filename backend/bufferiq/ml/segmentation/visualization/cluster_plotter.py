"""Cluster visualization utilities."""

from typing import Any, Dict, Optional

import numpy as np


class ClusterPlotter:
    """Plot clustering results."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize cluster plotter."""
        self.config = config or {}

    def prepare_2d_projection(
        self, feature_matrix: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Prepare 2D projection data for visualization.

        Args:
            feature_matrix: Feature matrix
            labels: Cluster labels

        Returns:
            Data for 2D visualization
        """
        from sklearn.decomposition import PCA

        # Reduce to 2D
        pca = PCA(n_components=2)
        projection = pca.fit_transform(feature_matrix)

        # Organize by cluster
        cluster_data = {}
        for label in set(labels):
            mask = labels == label
            cluster_data[int(label)] = {
                "x": projection[mask, 0].tolist(),
                "y": projection[mask, 1].tolist(),
                "size": int(np.sum(mask)),
            }

        return {
            "projection": projection.tolist(),
            "labels": labels.tolist(),
            "clusters": cluster_data,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        }

    def prepare_3d_projection(
        self, feature_matrix: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Prepare 3D projection data for visualization.

        Args:
            feature_matrix: Feature matrix
            labels: Cluster labels

        Returns:
            Data for 3D visualization
        """
        from sklearn.decomposition import PCA

        # Reduce to 3D
        pca = PCA(n_components=3)
        projection = pca.fit_transform(feature_matrix)

        # Organize by cluster
        cluster_data = {}
        for label in set(labels):
            mask = labels == label
            cluster_data[int(label)] = {
                "x": projection[mask, 0].tolist(),
                "y": projection[mask, 1].tolist(),
                "z": projection[mask, 2].tolist(),
                "size": int(np.sum(mask)),
            }

        return {
            "projection": projection.tolist(),
            "labels": labels.tolist(),
            "clusters": cluster_data,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        }

    def get_cluster_statistics(
        self, feature_matrix: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get statistics for each cluster.

        Args:
            feature_matrix: Feature matrix
            labels: Cluster labels

        Returns:
            Cluster statistics
        """
        statistics = {}

        for label in set(labels):
            mask = labels == label
            cluster_features = feature_matrix[mask]

            statistics[int(label)] = {
                "size": int(np.sum(mask)),
                "mean": cluster_features.mean(axis=0).tolist(),
                "std": cluster_features.std(axis=0).tolist(),
                "min": cluster_features.min(axis=0).tolist(),
                "max": cluster_features.max(axis=0).tolist(),
            }

        return statistics