"""
Saturation detector for hashtags.

Detects when hashtags are oversaturated.
"""

from typing import List, Tuple
from datetime import datetime, timedelta


class SaturationDetector:
    """
    Detect hashtag saturation.

    Identifies when hashtags are overused and losing effectiveness.

    Example:
```python
        detector = SaturationDetector()

        is_saturated = detector.check_saturation(
            hashtag="ai",
            recent_volume=5000,
            historical_avg=1500,
            engagement_trend=-0.15
        )

        if is_saturated:
            print("Hashtag is saturated - consider alternatives")
```
    """

    def __init__(
        self,
        volume_threshold: float = 3.0,
        engagement_decline_threshold: float = -0.10,
    ) -> None:
        """
        Initialize saturation detector.

        Args:
            volume_threshold: Volume increase multiplier to flag
            engagement_decline_threshold: Engagement decline % to flag
        """
        self.volume_threshold = volume_threshold
        self.engagement_decline_threshold = engagement_decline_threshold

    def check_saturation(
        self,
        hashtag: str,
        recent_volume: int,
        historical_avg: int,
        engagement_trend: float,
    ) -> bool:
        """
        Check if hashtag is saturated.

        Args:
            hashtag: Hashtag to check
            recent_volume: Recent usage volume
            historical_avg: Historical average volume
            engagement_trend: Engagement trend (-1 to 1)

        Returns:
            True if saturated
        """
        # Check volume spike
        volume_ratio = recent_volume / historical_avg if historical_avg > 0 else 1.0
        volume_spike = volume_ratio >= self.volume_threshold

        # Check engagement decline
        engagement_decline = engagement_trend <= self.engagement_decline_threshold

        # Saturated if both conditions met
        return volume_spike and engagement_decline

    def get_saturation_score(
        self,
        recent_volume: int,
        historical_avg: int,
        engagement_trend: float,
    ) -> float:
        """
        Get saturation score (0-100).

        Higher score = more saturated.

        Args:
            recent_volume: Recent volume
            historical_avg: Historical average
            engagement_trend: Engagement trend

        Returns:
            Saturation score (0-100)
        """
        # Volume factor
        volume_ratio = recent_volume / historical_avg if historical_avg > 0 else 1.0
        volume_factor = min(1.0, volume_ratio / 5.0)  # Cap at 5x

        # Engagement factor (invert and normalize)
        engagement_factor = max(0.0, -engagement_trend)

        # Combine (60% volume, 40% engagement decline)
        saturation = (volume_factor * 0.6 + engagement_factor * 0.4) * 100

        return max(0.0, min(100.0, saturation))