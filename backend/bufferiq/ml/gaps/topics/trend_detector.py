"""Trend detection for topics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TopicTrend:
    """Topic trend information."""

    topic_id: str
    trend_direction: str  # "rising", "stable", "falling"
    momentum: float  # 0-1
    velocity: float  # Rate of change
    significance: float  # Statistical significance (p-value)
    is_trending: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "topic_id": self.topic_id,
            "trend_direction": self.trend_direction,
            "momentum": self.momentum,
            "velocity": self.velocity,
            "significance": self.significance,
            "is_trending": self.is_trending,
        }


class TrendDetector:
    """
    Detect trending topics and analyze trend patterns.

    Uses statistical analysis to identify rising, stable, and falling trends.
    """

    def __init__(
        self,
        momentum_threshold: float = 0.6,
        significance_level: float = 0.05,
        min_data_points: int = 5,
    ):
        """
        Initialize trend detector.

        Args:
            momentum_threshold: Threshold for trending classification
            significance_level: P-value threshold for significance
            min_data_points: Minimum data points for trend analysis
        """
        self.momentum_threshold = momentum_threshold
        self.significance_level = significance_level
        self.min_data_points = min_data_points

    def detect_trend(
        self, topic_id: str, time_series: List[int], timestamps: List[datetime]
    ) -> TopicTrend:
        """
        Detect trend for a topic.

        Args:
            topic_id: Topic identifier
            time_series: Post counts over time
            timestamps: Corresponding timestamps

        Returns:
            Topic trend analysis
        """
        if len(time_series) < self.min_data_points:
            return TopicTrend(
                topic_id=topic_id,
                trend_direction="stable",
                momentum=0.0,
                velocity=0.0,
                significance=1.0,
                is_trending=False,
            )

        # Calculate trend using linear regression
        x = np.arange(len(time_series))
        y = np.array(time_series)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate momentum (normalized slope)
        momentum = self._calculate_momentum(slope, y)

        # Calculate velocity (rate of change)
        velocity = slope / (np.mean(y) + 1)  # Normalize by mean

        # Determine trend direction
        if p_value < self.significance_level:
            if slope > 0:
                trend_direction = "rising"
            elif slope < 0:
                trend_direction = "falling"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

        # Check if trending
        is_trending = (
            trend_direction == "rising"
            and momentum >= self.momentum_threshold
            and p_value < self.significance_level
        )

        return TopicTrend(
            topic_id=topic_id,
            trend_direction=trend_direction,
            momentum=round(momentum, 3),
            velocity=round(velocity, 3),
            significance=round(p_value, 4),
            is_trending=is_trending,
        )

    def _calculate_momentum(self, slope: float, values: np.ndarray) -> float:
        """
        Calculate momentum score (0-1).

        Args:
            slope: Regression slope
            values: Time series values

        Returns:
            Momentum score
        """
        if len(values) == 0:
            return 0.0

        # Normalize slope by data range
        data_range = np.ptp(values)  # Peak-to-peak
        if data_range == 0:
            return 0.0

        normalized_slope = abs(slope) / data_range

        # Clip to 0-1 range
        momentum = min(normalized_slope * 2, 1.0)

        return momentum

    def detect_seasonal_pattern(
        self, time_series: List[int], timestamps: List[datetime], period: int = 7
    ) -> Dict[str, float]:
        """
        Detect seasonal patterns (e.g., weekly cycles).

        Args:
            time_series: Post counts
            timestamps: Timestamps
            period: Expected period length (default: 7 days)

        Returns:
            Seasonality metrics
        """
        if len(time_series) < period * 2:
            return {"has_seasonality": False, "strength": 0.0}

        # Calculate autocorrelation at period lag
        if len(time_series) > period:
            autocorr = self._autocorrelation(time_series, period)
        else:
            autocorr = 0.0

        has_seasonality = autocorr > 0.5
        strength = min(abs(autocorr), 1.0)

        return {"has_seasonality": has_seasonality, "strength": round(strength, 3)}

    def _autocorrelation(self, series: List[int], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        arr = np.array(series)
        if len(arr) <= lag:
            return 0.0

        # Calculate correlation between series and lagged series
        y1 = arr[:-lag]
        y2 = arr[lag:]

        if len(y1) == 0 or len(y2) == 0:
            return 0.0

        correlation = np.corrcoef(y1, y2)[0, 1]

        return float(correlation) if not np.isnan(correlation) else 0.0