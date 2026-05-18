"""Trend momentum scoring."""

from typing import List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class MomentumScorer:
    """
    Calculate trend momentum scores.

    Measures the strength and acceleration of trends.
    """

    def calculate_momentum(
        self, time_series: List[float], window: int = 7
    ) -> float:
        """
        Calculate momentum score (0-1).

        Args:
            time_series: Values over time
            window: Window size for calculation

        Returns:
            Momentum score
        """
        if len(time_series) < window:
            return 0.0

        # Recent window
        recent = time_series[-window:]

        # Calculate rate of change
        roc = self._rate_of_change(recent)

        # Calculate acceleration
        accel = self._calculate_acceleration(recent)

        # Combine
        momentum = (abs(roc) * 0.6) + (abs(accel) * 0.4)

        # Normalize to 0-1
        momentum = min(momentum, 1.0)

        return round(momentum, 3)

    def _rate_of_change(self, values: List[float]) -> float:
        """Calculate rate of change."""
        if len(values) < 2:
            return 0.0

        first = values[0]
        last = values[-1]

        if first == 0:
            return 0.0

        roc = (last - first) / first

        return roc

    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate acceleration (second derivative)."""
        if len(values) < 3:
            return 0.0

        # Calculate first differences
        first_diff = np.diff(values)

        # Calculate second differences
        second_diff = np.diff(first_diff)

        # Average acceleration
        accel = float(np.mean(second_diff))

        return accel

    def identify_breakout(
        self, time_series: List[float], threshold: float = 0.7
    ) -> bool:
        """
        Identify breakout moments.

        Args:
            time_series: Values over time
            threshold: Momentum threshold for breakout

        Returns:
            True if breakout detected
        """
        momentum = self.calculate_momentum(time_series)

        return momentum >= threshold