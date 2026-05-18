"""Advanced trend detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    """Trend signal information."""

    topic: str
    signal_strength: float  # 0-1
    direction: str  # "rising", "falling", "stable"
    confidence: float  # 0-1
    velocity: float  # Rate of change
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "signal_strength": self.signal_strength,
            "direction": self.direction,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "detected_at": self.detected_at.isoformat(),
        }


class TrendDetector:
    """
    Detect trending topics with advanced statistical methods.

    Uses multiple indicators to identify emerging trends.
    """

    def __init__(
        self,
        min_data_points: int = 7,
        confidence_threshold: float = 0.7,
        velocity_threshold: float = 0.1,
    ):
        """
        Initialize trend detector.

        Args:
            min_data_points: Minimum data points for analysis
            confidence_threshold: Minimum confidence for trend signal
            velocity_threshold: Minimum velocity for trend detection
        """
        self.min_data_points = min_data_points
        self.confidence_threshold = confidence_threshold
        self.velocity_threshold = velocity_threshold

    def detect(
        self, topic: str, time_series: List[float], timestamps: List[datetime]
    ) -> TrendSignal:
        """
        Detect trend for a topic.

        Args:
            topic: Topic name
            time_series: Metric values over time
            timestamps: Corresponding timestamps

        Returns:
            Trend signal
        """
        if len(time_series) < self.min_data_points:
            return TrendSignal(
                topic=topic,
                signal_strength=0.0,
                direction="stable",
                confidence=0.0,
                velocity=0.0,
                detected_at=datetime.now(),
            )

        # Calculate trend components
        direction = self._calculate_direction(time_series)
        velocity = self._calculate_velocity(time_series)
        signal_strength = self._calculate_signal_strength(time_series)
        confidence = self._calculate_confidence(time_series)

        return TrendSignal(
            topic=topic,
            signal_strength=round(signal_strength, 3),
            direction=direction,
            confidence=round(confidence, 3),
            velocity=round(velocity, 3),
            detected_at=datetime.now(),
        )

    def _calculate_direction(self, series: List[float]) -> str:
        """Calculate trend direction."""
        if len(series) < 2:
            return "stable"

        # Linear regression
        x = np.arange(len(series))
        y = np.array(series)

        slope, _, _, p_value, _ = stats.linregress(x, y)

        # Check significance
        if p_value > 0.05:
            return "stable"

        if slope > self.velocity_threshold:
            return "rising"
        elif slope < -self.velocity_threshold:
            return "falling"
        else:
            return "stable"

    def _calculate_velocity(self, series: List[float]) -> float:
        """Calculate rate of change."""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = np.array(series)

        slope, _, _, _, _ = stats.linregress(x, y)

        # Normalize by mean
        mean_val = np.mean(y)
        if mean_val == 0:
            return 0.0

        velocity = slope / mean_val

        return float(velocity)

    def _calculate_signal_strength(self, series: List[float]) -> float:
        """Calculate signal strength (0-1)."""
        if len(series) < 2:
            return 0.0

        # Based on recent growth
        recent = series[-5:]
        early = series[:5] if len(series) >= 10 else series[:len(series)//2]

        recent_avg = np.mean(recent)
        early_avg = np.mean(early)

        if early_avg == 0:
            return 0.0

        growth_rate = (recent_avg - early_avg) / early_avg

        # Convert to 0-1 scale
        strength = min(abs(growth_rate), 1.0)

        return strength

    def _calculate_confidence(self, series: List[float]) -> float:
        """Calculate confidence in trend (0-1)."""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = np.array(series)

        _, _, r_value, p_value, _ = stats.linregress(x, y)

        # Confidence based on R² and p-value
        r_squared = r_value ** 2
        p_confidence = 1 - p_value

        # Combined confidence
        confidence = (r_squared + p_confidence) / 2

        return confidence