"""Data validation for segmentation."""

from typing import Any, List

import numpy as np

from bufferiq.ml.segmentation.types import AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    ValidationError,
    UnsupportedPlatformError,
)


class DataValidator:
    """Validate audience data before processing."""

    @staticmethod
    def validate_data_point(data_point: AudienceDataPoint) -> None:
        """
        Validate a single audience data point.

        Args:
            data_point: Data point to validate

        Raises:
            ValidationError: If data invalid
            UnsupportedPlatformError: If platform not supported
        """
        if not data_point.user_id:
            raise ValidationError("user_id cannot be empty")

        if data_point.platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(
                data_point.platform, SUPPORTED_PLATFORMS
            )

        if data_point.follower_count < 0:
            raise ValidationError("follower_count cannot be negative")

        if data_point.following_count < 0:
            raise ValidationError("following_count cannot be negative")

        if not (0 <= data_point.avg_engagement_rate <= 1):
            raise ValidationError("avg_engagement_rate must be between 0 and 1")

        if data_point.account_age_days < 0:
            raise ValidationError("account_age_days cannot be negative")

    @staticmethod
    def validate_feature_matrix(
        feature_matrix: np.ndarray, min_samples: int = 10
    ) -> None:
        """
        Validate feature matrix.

        Args:
            feature_matrix: Feature matrix to validate
            min_samples: Minimum required samples

        Raises:
            ValidationError: If matrix invalid
        """
        if feature_matrix.shape[0] < min_samples:
            raise ValidationError(
                f"Feature matrix has {feature_matrix.shape[0]} samples, "
                f"but minimum {min_samples} required"
            )

        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            raise ValidationError("Feature matrix cannot be empty")

        if np.isnan(feature_matrix).any():
            raise ValidationError("Feature matrix contains NaN values")

        if np.isinf(feature_matrix).any():
            raise ValidationError("Feature matrix contains infinite values")

    @staticmethod
    def validate_labels(
        labels: np.ndarray, n_clusters: int
    ) -> None:
        """
        Validate cluster labels.

        Args:
            labels: Cluster labels
            n_clusters: Expected number of clusters

        Raises:
            ValidationError: If labels invalid
        """
        if len(labels) == 0:
            raise ValidationError("Labels cannot be empty")

        unique_labels = len(set(labels))
        if unique_labels != n_clusters:
            raise ValidationError(
                f"Expected {n_clusters} clusters, got {unique_labels}"
            )