"""Clustering validation utilities."""

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from bufferiq.ml.segmentation.exceptions import ValidationError
from bufferiq.ml.segmentation.constants import MIN_SILHOUETTE_SCORE


class ClusteringValidator:
    """Validate clustering results."""

    @staticmethod
    def validate_result(
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        min_silhouette: float = MIN_SILHOUETTE_SCORE,
    ) -> bool:
        """
        Validate clustering result.

        Args:
            feature_matrix: Feature matrix
            labels: Cluster labels
            min_silhouette: Minimum silhouette score threshold

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if len(labels) != feature_matrix.shape[0]:
            raise ValidationError("Labels length must match feature matrix rows")

        if len(set(labels)) < 2 and min(labels) != -1:
            raise ValidationError("Must have at least 2 clusters")

        # Exclude noise points (-1) for silhouette calculation
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            sil_score = silhouette_score(
                feature_matrix[non_noise_mask], labels[non_noise_mask]
            )
            if sil_score < min_silhouette:
                raise ValidationError(
                    f"Silhouette score {sil_score:.3f} below minimum {min_silhouette:.3f}"
                )

        return True

    @staticmethod
    def check_cluster_sizes(labels: np.ndarray, min_size: int = 1) -> bool:
        """
        Check that all clusters meet minimum size.

        Args:
            labels: Cluster labels
            min_size: Minimum cluster size

        Returns:
            True if all clusters meet size requirement

        Raises:
            ValidationError: If cluster too small
        """
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label == -1:  # Skip noise
                continue
            if count < min_size:
                raise ValidationError(
                    f"Cluster {label} has {count} members, minimum {min_size}"
                )
        return True