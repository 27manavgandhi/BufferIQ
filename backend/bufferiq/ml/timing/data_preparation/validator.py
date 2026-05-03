"""Validate time-series data quality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class TimeSeriesValidator:
    """Validate time-series data quality."""

    def __init__(
        self,
        min_data_points: int = 100,
        max_std_devs: float = 3.0,
    ) -> None:
        """
        Initialize TimeSeriesValidator.

        Args:
            min_data_points: Minimum required data points
            max_std_devs: Threshold for outlier detection (std devs)
        """
        self.min_data_points = min_data_points
        self.max_std_devs = max_std_devs

    def validate(self, ts: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate time-series data.

        Args:
            ts: Time-series DataFrame

        Returns:
            Dict with validation results

        Example:
            >>> validator = TimeSeriesValidator(min_data_points=50)
            >>> result = validator.validate(ts)
            >>> assert result['is_valid']
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Check empty
        if ts.empty:
            results["is_valid"] = False
            results["errors"].append("Time-series is empty")
            return results

        # Check required columns
        if "timestamp" not in ts.columns:
            results["is_valid"] = False
            results["errors"].append("Missing 'timestamp' column")
            return results

        # Check minimum data points
        if len(ts) < self.min_data_points:
            results["warnings"].append(
                f"Only {len(ts)} data points (recommended: {self.min_data_points}+)"
            )

        # Check chronological order
        if not ts["timestamp"].is_monotonic_increasing:
            results["is_valid"] = False
            results["errors"].append("Timestamps are not in chronological order")

        # Check for duplicates
        if ts["timestamp"].duplicated().any():
            n_duplicates = ts["timestamp"].duplicated().sum()
            results["is_valid"] = False
            results["errors"].append(f"Found {n_duplicates} duplicate timestamps")

        # Check for extreme outliers in numeric columns
        numeric_cols = ts.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            outliers = self._detect_outliers(ts[col])
            if outliers.any():
                n_outliers = outliers.sum()
                results["warnings"].append(
                    f"Column '{col}' has {n_outliers} potential outliers "
                    f"(>{self.max_std_devs} std devs)"
                )

        # Add statistics
        results["stats"] = {
            "n_points": len(ts),
            "start_date": ts["timestamp"].min().isoformat(),
            "end_date": ts["timestamp"].max().isoformat(),
            "duration_days": (ts["timestamp"].max() - ts["timestamp"].min()).days,
        }

        return results

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using z-score method."""
        if series.std() == 0:
            return pd.Series([False] * len(series))

        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.max_std_devs