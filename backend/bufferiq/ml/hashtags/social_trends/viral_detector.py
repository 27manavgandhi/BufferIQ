"""
Viral content detector.

Detects viral content patterns and memes.
"""

from dataclasses import dataclass
from typing import List
from datetime import datetime


@dataclass
class ViralContent:
    """Detected viral content."""

    hashtag: str
    viral_score: float  # 0-100
    growth_rate: float
    detected_at: datetime
    pattern_type: str  # "exponential", "spike", "sustained"


class ViralContentDetector:
    """
    Detect viral content patterns.

    Example:
```python
        detector = ViralContentDetector()

        viral = detector.detect_viral(
            hashtag="ai",
            volume_history=[100, 200, 500, 1500, 5000]
        )

        if viral:
            print(f"Viral content detected!")
            print(f"  Score: {viral.viral_score:.1f}")
            print(f"  Pattern: {viral.pattern_type}")
```
    """

    def __init__(self, viral_threshold: float = 80.0) -> None:
        """
        Initialize viral detector.

        Args:
            viral_threshold: Minimum score to be considered viral
        """
        self.viral_threshold = viral_threshold

    def detect_viral(
        self,
        hashtag: str,
        volume_history: List[int],
    ) -> ViralContent | None:
        """
        Detect if content is going viral.

        Args:
            hashtag: Hashtag to check
            volume_history: Volume over time (chronological)

        Returns:
            Viral content if detected, None otherwise
        """
        if len(volume_history) < 3:
            return None

        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(volume_history)

        # Calculate viral score
        viral_score = self._calculate_viral_score(volume_history, growth_rate)

        if viral_score >= self.viral_threshold:
            # Determine pattern type
            pattern_type = self._determine_pattern(volume_history)

            return ViralContent(
                hashtag=hashtag,
                viral_score=viral_score,
                growth_rate=growth_rate,
                detected_at=datetime.now(),
                pattern_type=pattern_type,
            )

        return None

    def _calculate_growth_rate(self, volume_history: List[int]) -> float:
        """Calculate average growth rate."""
        if len(volume_history) < 2:
            return 0.0

        rates = []
        for i in range(1, len(volume_history)):
            if volume_history[i - 1] > 0:
                rate = (volume_history[i] - volume_history[i - 1]) / volume_history[
                    i - 1
                ]
                rates.append(rate)

        return sum(rates) / len(rates) if rates else 0.0

    def _calculate_viral_score(
        self, volume_history: List[int], growth_rate: float
    ) -> float:
        """
        Calculate viral score (0-100).

        Args:
            volume_history: Volume history
            growth_rate: Average growth rate

        Returns:
            Viral score
        """
        # Volume acceleration
        if len(volume_history) >= 2:
            recent_growth = (
                volume_history[-1] / volume_history[-2]
                if volume_history[-2] > 0
                else 1.0
            )
        else:
            recent_growth = 1.0

        # Combine growth rate and acceleration
        score = (growth_rate * 0.6 + (recent_growth - 1.0) * 0.4) * 100

        return max(0.0, min(100.0, score))

    def _determine_pattern(self, volume_history: List[int]) -> str:
        """Determine viral pattern type."""
        if len(volume_history) < 3:
            return "unknown"

        # Calculate growth rates
        rates = []
        for i in range(1, len(volume_history)):
            if volume_history[i - 1] > 0:
                rate = volume_history[i] / volume_history[i - 1]
                rates.append(rate)

        if not rates:
            return "unknown"

        avg_rate = sum(rates) / len(rates)

        # Check if accelerating
        if len(rates) >= 2 and rates[-1] > rates[0] * 1.5:
            return "exponential"

        # Check for spike (one big jump)
        if max(rates) > avg_rate * 2:
            return "spike"

        # Otherwise sustained
        return "sustained"