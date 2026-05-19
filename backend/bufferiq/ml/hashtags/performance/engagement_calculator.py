"""
Engagement calculator for hashtags.

Calculates various engagement metrics.
"""

from typing import Dict, List
import numpy as np


class EngagementCalculator:
    """
    Calculate engagement metrics for hashtags.

    Example:
```python
        calculator = EngagementCalculator()
        metrics = calculator.calculate(
            engagement_values=[100, 150, 120, 180, 140]
        )

        print(f"Average: {metrics['average']:.1f}")
        print(f"Median: {metrics['median']:.1f}")
        print(f"Std Dev: {metrics['std']:.1f}")
```
    """

    def calculate(self, engagement_values: List[float]) -> Dict[str, float]:
        """
        Calculate engagement metrics.

        Args:
            engagement_values: List of engagement values

        Returns:
            Dictionary of metrics
        """
        if not engagement_values:
            return {
                "average": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "percentile_25": 0.0,
                "percentile_75": 0.0,
                "percentile_90": 0.0,
            }

        arr = np.array(engagement_values)

        return {
            "average": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "percentile_25": float(np.percentile(arr, 25)),
            "percentile_75": float(np.percentile(arr, 75)),
            "percentile_90": float(np.percentile(arr, 90)),
        }

    def calculate_lift(
        self, with_hashtag: List[float], without_hashtag: List[float]
    ) -> float:
        """
        Calculate engagement lift.

        Args:
            with_hashtag: Engagement with hashtag
            without_hashtag: Engagement without hashtag

        Returns:
            Lift as decimal (0.25 = 25% increase)
        """
        if not with_hashtag or not without_hashtag:
            return 0.0

        avg_with = np.mean(with_hashtag)
        avg_without = np.mean(without_hashtag)

        if avg_without == 0:
            return 0.0

        lift = (avg_with - avg_without) / avg_without
        return float(lift)