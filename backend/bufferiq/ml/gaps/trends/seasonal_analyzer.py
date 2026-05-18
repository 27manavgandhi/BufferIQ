"""Seasonal pattern analysis."""

from datetime import datetime
from typing import Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SeasonalAnalyzer:
    """
    Analyze seasonal patterns in content performance.

    Identifies weekly, monthly, and yearly patterns.
    """

    def analyze_weekly_pattern(
        self, values: List[float], timestamps: List[datetime]
    ) -> Dict[str, float]:
        """
        Analyze weekly seasonality.

        Args:
            values: Metric values
            timestamps: Corresponding timestamps

        Returns:
            Day-of-week patterns
        """
        if len(values) < 7:
            return {}

        # Group by day of week
        day_values: Dict[int, List[float]] = {i: [] for i in range(7)}

        for value, timestamp in zip(values, timestamps):
            day_of_week = timestamp.weekday()
            day_values[day_of_week].append(value)

        # Calculate averages
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        patterns = {}

        for day_num, day_name in enumerate(day_names):
            if day_values[day_num]:
                patterns[day_name] = round(
                    float(np.mean(day_values[day_num])), 2
                )

        return patterns

    def analyze_monthly_pattern(
        self, values: List[float], timestamps: List[datetime]
    ) -> Dict[int, float]:
        """
        Analyze monthly seasonality.

        Args:
            values: Metric values
            timestamps: Corresponding timestamps

        Returns:
            Day-of-month patterns
        """
        if len(values) < 30:
            return {}

        # Group by day of month
        day_values: Dict[int, List[float]] = {}

        for value, timestamp in zip(values, timestamps):
            day = timestamp.day
            if day not in day_values:
                day_values[day] = []
            day_values[day].append(value)

        # Calculate averages
        patterns = {}
        for day, vals in day_values.items():
            if vals:
                patterns[day] = round(float(np.mean(vals)), 2)

        return patterns

    def detect_seasonality(
        self, time_series: List[float], period: int = 7
    ) -> Dict[str, any]:
        """
        Detect seasonal patterns.

        Args:
            time_series: Time series values
            period: Expected period (e.g., 7 for weekly)

        Returns:
            Seasonality detection results
        """
        if len(time_series) < period * 2:
            return {"has_seasonality": False, "strength": 0.0}

        # Calculate autocorrelation at period lag
        autocorr = self._autocorrelation(time_series, period)

        has_seasonality = autocorr > 0.5
        strength = min(abs(autocorr), 1.0)

        return {
            "has_seasonality": has_seasonality,
            "strength": round(strength, 3),
            "period": period,
        }

    def _autocorrelation(self, series: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        arr = np.array(series)

        if len(arr) <= lag:
            return 0.0

        # Pearson correlation between series and lagged series
        y1 = arr[:-lag]
        y2 = arr[lag:]

        if len(y1) == 0 or len(y2) == 0:
            return 0.0

        corr = np.corrcoef(y1, y2)[0, 1]

        return float(corr) if not np.isnan(corr) else 0.0