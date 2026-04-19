"""Deep-dive performance analysis."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """Deep-dive performance analysis."""

    def __init__(self) -> None:
        """
        Initialize analyzer.

        Example:
            >>> analyzer = PerformanceAnalyzer()
            >>> bias = analyzer.detect_systematic_bias(y_true, y_pred)
        """
        pass

    def analyze_performance_by_percentile(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentiles: List[int] = [25, 50, 75, 90, 95],
    ) -> pd.DataFrame:
        """
        Analyze performance at different percentiles.

        Args:
            y_true: True values
            y_pred: Predicted values
            percentiles: Percentiles to analyze

        Returns:
            DataFrame with percentile analysis

        Example:
            >>> percentile_perf = analyzer.analyze_performance_by_percentile(
            ...     y_true, y_pred, [25, 50, 75, 90, 95]
            ... )
        """
        from sklearn.metrics import mean_absolute_error, r2_score

        results = []

        # Calculate percentile thresholds
        thresholds = np.percentile(y_true, percentiles)

        for i, percentile in enumerate(percentiles):
            # Get samples in this percentile range
            if i == 0:
                mask = y_true <= thresholds[i]
            else:
                mask = (y_true > thresholds[i - 1]) & (y_true <= thresholds[i])

            if mask.sum() > 1:
                y_true_range = y_true[mask]
                y_pred_range = y_pred[mask]

                mae = mean_absolute_error(y_true_range, y_pred_range)
                r2 = r2_score(y_true_range, y_pred_range)

                results.append(
                    {
                        "percentile": f"0-{percentile}" if i == 0 else f"{percentiles[i-1]}-{percentile}",
                        "count": int(mask.sum()),
                        "mae": float(mae),
                        "r2": float(r2),
                        "mean_actual": float(np.mean(y_true_range)),
                    }
                )

        return pd.DataFrame(results)

    def analyze_performance_by_prediction_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_metric: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Analyze performance by prediction confidence.

        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_metric: Optional confidence scores

        Returns:
            DataFrame with confidence analysis

        Example:
            >>> confidence_perf = analyzer.analyze_performance_by_prediction_confidence(
            ...     y_true, y_pred
            ... )
        """
        from sklearn.metrics import mean_absolute_error

        # If no confidence metric provided, use prediction magnitude
        if confidence_metric is None:
            confidence_metric = np.abs(y_pred)

        # Divide into confidence quartiles
        quartiles = np.percentile(confidence_metric, [25, 50, 75])

        results = []

        for i, label in enumerate(["Low", "Medium-Low", "Medium-High", "High"]):
            if i == 0:
                mask = confidence_metric <= quartiles[0]
            elif i == 3:
                mask = confidence_metric > quartiles[2]
            else:
                mask = (confidence_metric > quartiles[i - 1]) & (
                    confidence_metric <= quartiles[i]
                )

            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])

                results.append(
                    {
                        "confidence_level": label,
                        "count": int(mask.sum()),
                        "mae": float(mae),
                    }
                )

        return pd.DataFrame(results)

    def detect_systematic_bias(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect systematic prediction bias.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dict with bias analysis

        Example:
            >>> bias = analyzer.detect_systematic_bias(y_true, y_pred)
            >>> print(f"Overall bias: {bias['overall_bias']:.4f}")
        """
        errors = y_true - y_pred

        # Overall bias
        overall_bias = float(np.mean(errors))

        # Over/underestimation rates
        overestimation_rate = float(np.mean(errors < 0))
        underestimation_rate = float(np.mean(errors > 0))

        # Bias by value range
        low_mask = y_true < np.percentile(y_true, 33)
        mid_mask = (y_true >= np.percentile(y_true, 33)) & (
            y_true < np.percentile(y_true, 67)
        )
        high_mask = y_true >= np.percentile(y_true, 67)

        bias_by_range = {
            "low_values": float(np.mean(errors[low_mask])) if low_mask.sum() > 0 else 0.0,
            "mid_values": float(np.mean(errors[mid_mask])) if mid_mask.sum() > 0 else 0.0,
            "high_values": float(np.mean(errors[high_mask])) if high_mask.sum() > 0 else 0.0,
        }

        return {
            "overall_bias": overall_bias,
            "overestimation_rate": overestimation_rate,
            "underestimation_rate": underestimation_rate,
            "bias_by_range": bias_by_range,
        }

    def analyze_error_correlation_with_features(
        self, errors: np.ndarray, features: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Find features most correlated with prediction errors.

        Args:
            errors: Prediction errors
            features: Feature DataFrame
            top_n: Top N features to return

        Returns:
            DataFrame with error correlations

        Example:
            >>> error_corr = analyzer.analyze_error_correlation_with_features(
            ...     errors, X_test, top_n=10
            ... )
        """
        correlations = []

        for col in features.columns:
            if pd.api.types.is_numeric_dtype(features[col]):
                corr = np.corrcoef(features[col], np.abs(errors))[0, 1]

                if not np.isnan(corr):
                    correlations.append(
                        {"feature": col, "correlation": float(abs(corr))}
                    )

        # Sort by absolute correlation
        corr_df = pd.DataFrame(correlations)

        if not corr_df.empty:
            corr_df = corr_df.sort_values("correlation", ascending=False)
            return corr_df.head(top_n)
        else:
            return pd.DataFrame()