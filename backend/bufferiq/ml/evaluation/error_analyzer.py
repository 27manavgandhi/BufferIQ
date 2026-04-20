"""Error analysis and failure mode detection."""

from typing import Any

import numpy as np
import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ErrorAnalyzer:
    """Analyze prediction errors and failure modes."""

    def __init__(self) -> None:
        """
        Initialize error analyzer.

        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> error_classes = analyzer.classify_errors(y_true, y_pred)
        """
        pass

    def classify_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: dict[str, float] = {"low": 0.1, "medium": 0.2, "high": 0.3},
    ) -> dict[str, int]:
        """
        Classify errors into categories.

        Args:
            y_true: True values
            y_pred: Predicted values
            thresholds: Error thresholds

        Returns:
            Dict with error counts

        Example:
            >>> error_classes = analyzer.classify_errors(y_true, y_pred)
            >>> print(error_classes)
        """
        abs_errors = np.abs(y_true - y_pred)

        classification = {
            "low_error": int(np.sum(abs_errors <= thresholds["low"])),
            "medium_error": int(
                np.sum(
                    (abs_errors > thresholds["low"])
                    & (abs_errors <= thresholds["medium"])
                )
            ),
            "high_error": int(
                np.sum(
                    (abs_errors > thresholds["medium"])
                    & (abs_errors <= thresholds["high"])
                )
            ),
            "very_high_error": int(np.sum(abs_errors > thresholds["high"])),
        }

        logger.info(f"Error classification: {classification}")

        return classification

    def identify_failure_modes(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        features: pd.DataFrame,
        error_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Identify common patterns in high-error predictions.

        Args:
            y_true: True values
            y_pred: Predicted values
            features: Feature DataFrame
            error_threshold: Threshold for high error

        Returns:
            List of failure modes

        Example:
            >>> failure_modes = analyzer.identify_failure_modes(
            ...     y_test, predictions, X_test, error_threshold=0.3
            ... )
        """
        abs_errors = np.abs(y_true.values - y_pred)
        high_error_mask = abs_errors > error_threshold

        if high_error_mask.sum() == 0:
            return []

        failure_modes = []

        # Analyze by text length
        if "text_length" in features.columns:
            high_error_features = features[high_error_mask]
            avg_text_length = high_error_features["text_length"].mean()
            overall_avg = features["text_length"].mean()

            if abs(avg_text_length - overall_avg) > overall_avg * 0.2:
                failure_modes.append(
                    {
                        "description": f"High errors on {'long' if avg_text_length > overall_avg else 'short'} text",
                        "count": int(high_error_mask.sum()),
                        "avg_error": float(np.mean(abs_errors[high_error_mask])),
                        "avg_text_length": float(avg_text_length),
                    }
                )

        # Analyze by hashtag count
        if "hashtag_count" in features.columns:
            high_error_features = features[high_error_mask]
            avg_hashtags = high_error_features["hashtag_count"].mean()

            if avg_hashtags > 2:
                failure_modes.append(
                    {
                        "description": "High errors on posts with many hashtags",
                        "count": int(high_error_mask.sum()),
                        "avg_error": float(np.mean(abs_errors[high_error_mask])),
                        "avg_hashtags": float(avg_hashtags),
                    }
                )

        return failure_modes

    def analyze_error_by_feature_ranges(
        self,
        errors: np.ndarray,
        features: pd.DataFrame,
        feature_name: str,
        n_bins: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze error distribution across feature value ranges.

        Args:
            errors: Prediction errors
            features: Feature DataFrame
            feature_name: Feature to analyze
            n_bins: Number of bins

        Returns:
            DataFrame with error by feature range

        Example:
            >>> error_by_length = analyzer.analyze_error_by_feature_ranges(
            ...     errors, X_test, "text_length", n_bins=5
            ... )
        """
        if feature_name not in features.columns:
            raise ValueError(f"Feature '{feature_name}' not found")

        feature_values = features[feature_name].values

        # Create bins
        bins = pd.qcut(feature_values, q=n_bins, duplicates="drop")

        # Calculate error stats by bin
        results = []

        for bin_label in bins.cat.categories:
            mask = bins == bin_label
            if mask.sum() > 0:
                results.append(
                    {
                        "range": str(bin_label),
                        "count": int(mask.sum()),
                        "mean_abs_error": float(np.mean(np.abs(errors[mask]))),
                        "std_error": float(np.std(errors[mask])),
                    }
                )

        return pd.DataFrame(results)
