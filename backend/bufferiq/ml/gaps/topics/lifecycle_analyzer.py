"""Topic lifecycle stage analysis."""

from datetime import datetime
from enum import Enum
from typing import Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class LifecycleStage(Enum):
    """Topic lifecycle stages."""

    EMERGING = "emerging"  # New, growing fast
    GROWING = "growing"  # Steady increase
    MATURE = "mature"  # Stable, high volume
    DECLINING = "declining"  # Decreasing


class LifecycleAnalyzer:
    """
    Analyze topic lifecycle stages.

    Determines if topics are emerging, growing, mature, or declining
    based on temporal patterns.
    """

    def __init__(self, growth_threshold: float = 0.2, decline_threshold: float = -0.15):
        """
        Initialize lifecycle analyzer.

        Args:
            growth_threshold: Growth rate threshold for growing stage
            decline_threshold: Decline rate threshold for declining stage
        """
        self.growth_threshold = growth_threshold
        self.decline_threshold = decline_threshold

    def determine_stage(
        self, post_counts: Dict[datetime, int], time_points: List[datetime]
    ) -> str:
        """
        Determine lifecycle stage from temporal data.

        Args:
            post_counts: Posts per time period
            time_points: Corresponding dates

        Returns:
            Stage: "emerging", "growing", "mature", "declining"
        """
        if len(post_counts) < 3:
            return LifecycleStage.EMERGING.value

        # Convert to time series
        sorted_times = sorted(time_points)
        counts = [post_counts.get(t, 0) for t in sorted_times]

        # Calculate recent trend
        recent_counts = counts[-5:] if len(counts) >= 5 else counts
        trend = self._calculate_trend(recent_counts)

        # Calculate volume level
        avg_count = np.mean(counts)
        max_count = np.max(counts)

        # Determine stage based on trend and volume
        if avg_count < 5:
            # Low volume
            if trend > self.growth_threshold:
                return LifecycleStage.EMERGING.value
            else:
                return LifecycleStage.DECLINING.value

        elif trend > self.growth_threshold:
            # Growing
            return LifecycleStage.GROWING.value

        elif trend < self.decline_threshold:
            # Declining
            return LifecycleStage.DECLINING.value

        else:
            # Stable
            return LifecycleStage.MATURE.value

    def _calculate_trend(self, counts: List[int]) -> float:
        """
        Calculate trend coefficient.

        Args:
            counts: Post counts over time

        Returns:
            Trend coefficient (-1 to 1)
        """
        if len(counts) < 2:
            return 0.0

        # Simple linear regression
        x = np.arange(len(counts))
        y = np.array(counts)

        # Handle zero variance
        if np.std(y) == 0:
            return 0.0

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Normalize by mean
        mean_val = np.mean(y)
        if mean_val == 0:
            return 0.0

        normalized_slope = slope / mean_val

        return float(np.clip(normalized_slope, -1, 1))

    def calculate_maturity_score(
        self, post_counts: Dict[datetime, int], time_span_days: int
    ) -> float:
        """
        Calculate topic maturity score (0-1).

        Args:
            post_counts: Post counts over time
            time_span_days: Total time span in days

        Returns:
            Maturity score
        """
        if not post_counts:
            return 0.0

        # Factors for maturity:
        # 1. Time span (longer = more mature)
        # 2. Consistency (stable posting = more mature)
        # 3. Volume (higher = more mature)

        counts = list(post_counts.values())

        # Time factor
        time_factor = min(time_span_days / 180, 1.0)  # Max at 6 months

        # Consistency factor (inverse of coefficient of variation)
        if np.mean(counts) > 0:
            cv = np.std(counts) / np.mean(counts)
            consistency_factor = 1 / (1 + cv)
        else:
            consistency_factor = 0.0

        # Volume factor
        volume_factor = min(np.mean(counts) / 20, 1.0)  # Max at 20 posts/period

        # Weighted average
        maturity = (time_factor * 0.4) + (consistency_factor * 0.3) + (volume_factor * 0.3)

        return round(maturity, 3)