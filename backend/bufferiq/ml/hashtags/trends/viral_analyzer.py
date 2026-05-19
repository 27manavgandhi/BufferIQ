"""
Viral content analyzer for hashtags.

Identifies viral patterns in hashtag usage.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ViralPattern:
    """Detected viral pattern."""

    pattern_type: str  # "exponential", "spike", "sustained"
    confidence: float  # 0-1
    growth_rate: float
    acceleration: float  # Rate of change in growth


class ViralAnalyzer:
    """
    Analyze viral patterns in hashtag usage.

    Example:
```python
        analyzer = ViralAnalyzer()
        pattern = analyzer.detect_viral_pattern(
            volume_history=[100, 150, 300, 800, 2000]
        )

        if pattern:
            print(f"Viral pattern: {pattern.pattern_type}")
            print(f"Confidence: {pattern.confidence:.2f}")
```
    """

    def __init__(self, viral_threshold: float = 2.0) -> None:
        """
        Initialize viral analyzer.

        Args:
            viral_threshold: Growth multiplier to consider viral
        """
        self.viral_threshold = viral_threshold

    def detect_viral_pattern(
        self, volume_history: List[int]
    ) -> ViralPattern | None:
        """
        Detect viral pattern in volume history.

        Args:
            volume_history: Chronological volume history

        Returns:
            Detected pattern or None
        """
        if len(volume_history) < 3:
            return None

        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(volume_history)):
            if volume_history[i - 1] > 0:
                rate = volume_history[i] / volume_history[i - 1]
                growth_rates.append(rate)

        if not growth_rates:
            return None

        avg_growth = np.mean(growth_rates)

        # Check if viral (exponential growth)
        if avg_growth >= self.viral_threshold:
            # Calculate acceleration
            if len(growth_rates) >= 2:
                acceleration = growth_rates[-1] / growth_rates[0]
            else:
                acceleration = 1.0

            # Determine pattern type
            if acceleration > 1.5:
                pattern_type = "exponential"
            elif max(growth_rates) > avg_growth * 2:
                pattern_type = "spike"
            else:
                pattern_type = "sustained"

            # Confidence based on consistency
            std_growth = np.std(growth_rates)
            consistency = 1.0 / (1.0 + std_growth)

            return ViralPattern(
                pattern_type=pattern_type,
                confidence=float(consistency),
                growth_rate=float(avg_growth),
                acceleration=float(acceleration),
            )

        return None