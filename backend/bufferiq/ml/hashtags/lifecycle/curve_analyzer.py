"""
Growth/decline curve analyzer.

Analyzes hashtag growth and decline curves.
"""

from typing import List, Tuple
import numpy as np


class CurveAnalyzer:
    """
    Analyze hashtag growth/decline curves.

    Example:
```python
        analyzer = CurveAnalyzer()

        curve_type = analyzer.analyze_curve(
            volume_history=[10, 30, 100, 300, 500, 400, 300]
        )

        print(f"Curve type: {curve_type}")
```
    """

    def analyze_curve(self, volume_history: List[int]) -> str:
        """
        Analyze volume curve type.

        Args:
            volume_history: Volume over time

        Returns:
            Curve type: "exponential", "linear", "logarithmic", "bell", "declining"
        """
        if len(volume_history) < 3:
            return "insufficient_data"

        # Find peak
        peak_idx = volume_history.index(max(volume_history))

        # Analyze pre-peak growth
        if peak_idx > 0:
            pre_peak = volume_history[: peak_idx + 1]
            growth_type = self._analyze_growth(pre_peak)
        else:
            growth_type = "unknown"

        # Analyze post-peak
        if peak_idx < len(volume_history) - 1:
            post_peak = volume_history[peak_idx:]
            is_declining = all(
                post_peak[i] >= post_peak[i + 1] for i in range(len(post_peak) - 1)
            )
        else:
            is_declining = False

        # Determine overall curve
        if is_declining and growth_type != "unknown":
            return "bell"
        elif is_declining:
            return "declining"
        else:
            return growth_type

    def _analyze_growth(self, volumes: List[int]) -> str:
        """Analyze growth pattern."""
        if len(volumes) < 2:
            return "unknown"

        # Calculate growth rates
        rates = []
        for i in range(1, len(volumes)):
            if volumes[i - 1] > 0:
                rate = volumes[i] / volumes[i - 1]
                rates.append(rate)

        if not rates:
            return "unknown"

        avg_rate = np.mean(rates)

        # Exponential: increasing growth rate
        if avg_rate > 2.0:
            return "exponential"
        # Linear: steady growth
        elif 1.2 <= avg_rate <= 2.0:
            return "linear"
        # Logarithmic: decreasing growth rate
        else:
            return "logarithmic"

    def calculate_growth_rate(self, volume_history: List[int]) -> float:
        """
        Calculate average growth rate.

        Args:
            volume_history: Volume over time

        Returns:
            Average growth rate
        """
        if len(volume_history) < 2:
            return 0.0

        rates = []
        for i in range(1, len(volume_history)):
            if volume_history[i - 1] > 0:
                rate = (volume_history[i] - volume_history[i - 1]) / volume_history[
                    i - 1
                ]
                rates.append(rate)

        return np.mean(rates) if rates else 0.0