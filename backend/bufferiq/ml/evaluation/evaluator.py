"""Comprehensive model evaluation with multiple metrics."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self, output_dir: str = "outputs/evaluations") -> None:
        """
        Initialize evaluator.

        Args:
            output_dir: Directory for evaluation outputs

        Example:
            >>> evaluator = ModelEvaluator()
            >>> metrics = evaluator.calculate_metrics(y_true, y_pred)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "residual_plots").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "error_analysis").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate all regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dict with all metrics

        Example:
            >>> metrics = evaluator.calculate_metrics(y_true, y_pred)
            >>> print(f"R²: {metrics['r2']:.4f}")
        """
        # Handle edge cases
        if len(y_true) == 0:
            raise ValueError("Empty input arrays")

        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have same length")

        # Calculate errors
        errors = y_true - y_pred
        squared_errors = errors**2

        # Calculate metrics
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "max_error": float(max_error(y_true, y_pred)),
            "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mean_squared_error": float(mean_squared_error(y_true, y_pred)),
        }

        # MAPE only if no zeros in y_true
        if (y_true != 0).all():
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        else:
            metrics["mape"] = 0.0

        return metrics

    def evaluate_by_platform(
        self, y_true: pd.Series, y_pred: np.ndarray, platforms: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate metrics per platform.

        Args:
            y_true: True values
            y_pred: Predicted values
            platforms: Platform labels

        Returns:
            DataFrame with platform-wise metrics

        Example:
            >>> platform_metrics = evaluator.evaluate_by_platform(
            ...     y_test, predictions, platforms
            ... )
        """
        if len(y_true) != len(platforms):
            raise ValueError("y_true and platforms must have same length")

        results = []

        for platform in platforms.unique():
            mask = platforms == platform
            platform_y_true = y_true[mask]
            platform_y_pred = y_pred[mask]

            if len(platform_y_true) > 0:
                metrics = self.calculate_metrics(
                    platform_y_true.values, platform_y_pred
                )
                metrics["platform"] = platform
                metrics["count"] = int(len(platform_y_true))
                results.append(metrics)

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ["platform", "count", "mae", "rmse", "r2", "mape"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols + [c for c in df.columns if c not in cols]]

        return df

    def evaluate_by_time_period(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        timestamps: pd.Series,
        period: str = "month",
    ) -> pd.DataFrame:
        """
        Evaluate metrics over time periods.

        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Timestamps for each prediction
            period: 'day', 'week', 'month', 'quarter'

        Returns:
            DataFrame with temporal metrics

        Example:
            >>> temporal_metrics = evaluator.evaluate_by_time_period(
            ...     y_test, predictions, timestamps, period='month'
            ... )
        """
        if len(y_true) != len(timestamps):
            raise ValueError("y_true and timestamps must have same length")

        # Create DataFrame for easier grouping
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "timestamp": timestamps})

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by period
        if period == "day":
            df["period"] = df["timestamp"].dt.date
        elif period == "week":
            df["period"] = df["timestamp"].dt.to_period("W")
        elif period == "month":
            df["period"] = df["timestamp"].dt.to_period("M")
        elif period == "quarter":
            df["period"] = df["timestamp"].dt.to_period("Q")
        else:
            raise ValueError(f"Invalid period: {period}")

        results = []

        for period_value, group in df.groupby("period"):
            if len(group) > 1:  # Need at least 2 samples
                metrics = self.calculate_metrics(
                    group["y_true"].values, group["y_pred"].values
                )
                metrics["period"] = str(period_value)
                metrics["count"] = int(len(group))
                results.append(metrics)

        return pd.DataFrame(results)

    def evaluate_by_content_type(
        self, y_true: pd.Series, y_pred: np.ndarray, features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate by content characteristics.

        Args:
            y_true: True values
            y_pred: Predicted values
            features: Feature DataFrame

        Returns:
            DataFrame with content-type metrics

        Example:
            >>> content_metrics = evaluator.evaluate_by_content_type(
            ...     y_test, predictions, X_test
            ... )
        """
        results = []

        # Has URL vs No URL
        if "has_url" in features.columns:
            for has_url in [True, False]:
                mask = features["has_url"] == has_url
                if mask.sum() > 0:
                    metrics = self.calculate_metrics(y_true[mask].values, y_pred[mask])
                    metrics["content_type"] = "Has URL" if has_url else "No URL"
                    metrics["count"] = int(mask.sum())
                    results.append(metrics)

        # Has Hashtag vs No Hashtag
        if "hashtag_count" in features.columns:
            for has_hashtag in [True, False]:
                mask = (
                    features["hashtag_count"] > 0
                    if has_hashtag
                    else features["hashtag_count"] == 0
                )
                if mask.sum() > 0:
                    metrics = self.calculate_metrics(y_true[mask].values, y_pred[mask])
                    metrics["content_type"] = (
                        "Has Hashtag" if has_hashtag else "No Hashtag"
                    )
                    metrics["count"] = int(mask.sum())
                    results.append(metrics)

        # Text length categories
        if "text_length" in features.columns:
            text_length = features["text_length"]
            # Define bins
            bins = [0, 100, 200, float("inf")]
            labels = ["Short (0-100)", "Medium (100-200)", "Long (200+)"]

            for i, label in enumerate(labels):
                mask = (text_length >= bins[i]) & (text_length < bins[i + 1])
                if mask.sum() > 0:
                    metrics = self.calculate_metrics(y_true[mask].values, y_pred[mask])
                    metrics["content_type"] = label
                    metrics["count"] = int(mask.sum())
                    results.append(metrics)

        return pd.DataFrame(results) if results else pd.DataFrame()

    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate residuals.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Residuals array

        Example:
            >>> residuals = evaluator.calculate_residuals(y_true, y_pred)
        """
        return y_true - y_pred

    def identify_worst_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        features: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Identify worst predictions for error analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            features: Feature DataFrame
            top_n: Number of worst predictions to return

        Returns:
            DataFrame with worst predictions

        Example:
            >>> worst = evaluator.identify_worst_predictions(
            ...     y_test, predictions, X_test, top_n=20
            ... )
        """
        # Calculate errors
        errors = np.abs(y_true.values - y_pred)

        # Create DataFrame
        worst_df = pd.DataFrame(
            {
                "actual": y_true.values,
                "predicted": y_pred,
                "error": errors,
                "abs_error": errors,
            }
        )

        # Add some features
        feature_cols = ["text_length", "hashtag_count", "has_url"]
        for col in feature_cols:
            if col in features.columns:
                worst_df[col] = features[col].values

        # Sort by error
        worst_df = worst_df.sort_values("abs_error", ascending=False)

        return worst_df.head(top_n)

    def generate_evaluation_summary(
        self,
        trainer: BaseTrainer,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        platforms: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive evaluation summary.

        Args:
            trainer: Trained model
            X_test: Test features
            y_test: Test targets
            platforms: Platform labels
            timestamps: Timestamps

        Returns:
            Dict with comprehensive evaluation

        Example:
            >>> summary = evaluator.generate_evaluation_summary(
            ...     trainer, X_test, y_test, platforms, timestamps
            ... )
        """
        # Generate predictions
        y_pred = trainer.predict(X_test)

        # Overall metrics
        overall_metrics = self.calculate_metrics(y_test.values, y_pred)

        summary: dict[str, Any] = {"overall_metrics": overall_metrics}

        # Platform metrics
        if platforms is not None:
            summary["platform_metrics"] = self.evaluate_by_platform(
                y_test, y_pred, platforms
            )

        # Temporal metrics
        if timestamps is not None:
            summary["temporal_metrics"] = self.evaluate_by_time_period(
                y_test, y_pred, timestamps, period="month"
            )

        # Content type metrics
        summary["content_type_metrics"] = self.evaluate_by_content_type(
            y_test, y_pred, X_test
        )

        # Worst predictions
        summary["worst_predictions"] = self.identify_worst_predictions(
            y_test, y_pred, X_test, top_n=20
        )

        logger.info(
            f"Generated evaluation summary: R²={overall_metrics['r2']:.4f}, MAE={overall_metrics['mae']:.4f}"
        )

        return summary
