"""Model diagnostics and health checks."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ModelDiagnostics:
    """Diagnose model health and potential issues."""

    def __init__(self) -> None:
        """
        Initialize diagnostics.

        Example:
            >>> diagnostics = ModelDiagnostics()
            >>> overfitting = diagnostics.check_overfitting(train_metrics, val_metrics)
        """
        pass

    def check_overfitting(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        Check for overfitting.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            threshold: R² gap threshold

        Returns:
            Dict with overfitting analysis

        Example:
            >>> overfitting = diagnostics.check_overfitting(
            ...     {"r2": 0.95}, {"r2": 0.75}, threshold=0.1
            ... )
        """
        train_r2 = train_metrics.get("r2", 0.0)
        val_r2 = val_metrics.get("r2", 0.0)

        gap = train_r2 - val_r2
        is_overfitting = gap > threshold

        # Determine severity
        if gap > 0.2:
            severity = "high"
        elif gap > 0.1:
            severity = "medium"
        else:
            severity = "low"

        result = {
            "is_overfitting": is_overfitting,
            "train_val_gap": float(gap),
            "severity": severity,
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
        }

        if is_overfitting:
            logger.warning(
                f"Overfitting detected: train R²={train_r2:.4f}, val R²={val_r2:.4f}, gap={gap:.4f}"
            )

        return result

    def check_underfitting(
        self, metrics: dict[str, float], min_r2: float = 0.5
    ) -> dict[str, Any]:
        """
        Check for underfitting.

        Args:
            metrics: Model metrics
            min_r2: Minimum acceptable R²

        Returns:
            Dict with underfitting analysis

        Example:
            >>> underfitting = diagnostics.check_underfitting({"r2": 0.45}, min_r2=0.5)
        """
        r2 = metrics.get("r2", 0.0)
        is_underfitting = r2 < min_r2

        # Determine severity
        if r2 < 0.3:
            severity = "high"
        elif r2 < 0.5:
            severity = "medium"
        else:
            severity = "low"

        result = {
            "is_underfitting": is_underfitting,
            "r2_score": float(r2),
            "severity": severity,
            "min_r2": min_r2,
        }

        if is_underfitting:
            logger.warning(f"Underfitting detected: R²={r2:.4f} < {min_r2}")

        return result

    def check_residual_patterns(self, residuals: np.ndarray) -> dict[str, Any]:
        """
        Check for patterns in residuals.

        Args:
            residuals: Model residuals

        Returns:
            Dict with residual diagnostics

        Example:
            >>> residual_check = diagnostics.check_residual_patterns(residuals)
        """
        # Test for zero mean
        mean_residual = float(np.mean(residuals))
        mean_close_to_zero = abs(mean_residual) < 0.01

        # Test for constant variance (Levene's test on halves)
        mid_point = len(residuals) // 2
        first_half = residuals[:mid_point]
        second_half = residuals[mid_point:]

        if len(first_half) > 1 and len(second_half) > 1:
            _, p_value_variance = stats.levene(first_half, second_half)
            constant_variance = p_value_variance > 0.05
        else:
            p_value_variance = 1.0
            constant_variance = True

        # Test for normality (Shapiro-Wilk test)
        if len(residuals) > 3:
            _, p_value_normality = stats.shapiro(
                residuals[:5000]
            )  # Limit to 5000 samples
            is_normal = p_value_normality > 0.05
        else:
            p_value_normality = 1.0
            is_normal = True

        result = {
            "mean_residual": mean_residual,
            "mean_close_to_zero": mean_close_to_zero,
            "constant_variance": constant_variance,
            "variance_test_p_value": float(p_value_variance),
            "is_normal": is_normal,
            "normality_test_p_value": float(p_value_normality),
            "healthy_residuals": mean_close_to_zero and constant_variance and is_normal,
        }

        return result

    def check_feature_importance_concentration(
        self, importance: pd.DataFrame, threshold: float = 0.8
    ) -> dict[str, Any]:
        """
        Check if importance is concentrated in few features.

        Args:
            importance: Feature importance DataFrame
            threshold: Concentration threshold

        Returns:
            Dict with concentration analysis

        Example:
            >>> concentration = diagnostics.check_feature_importance_concentration(
            ...     importance, threshold=0.8
            ... )
        """
        # Calculate cumulative importance
        total_importance = importance["importance"].sum()
        cumulative = importance["importance"].cumsum() / total_importance

        # Find how many features to reach threshold
        features_to_threshold = int((cumulative >= threshold).sum())

        is_concentrated = features_to_threshold < len(importance) * 0.2

        result = {
            "is_concentrated": is_concentrated,
            "features_to_threshold": features_to_threshold,
            "total_features": len(importance),
            "concentration_ratio": float(features_to_threshold / len(importance)),
        }

        if is_concentrated:
            logger.warning(
                f"Feature importance concentrated: {features_to_threshold} features account for {threshold*100}%"
            )

        return result
