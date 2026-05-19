"""
Hashtag effectiveness scorer.

Multi-factor scoring of hashtag effectiveness.
"""

from typing import Dict, List
import numpy as np


class EffectivenessScorer:
    """
    Calculate hashtag effectiveness scores.

    Combines multiple factors:
    - Engagement rate
    - Reach amplification
    - Competition level
    - Trend momentum
    - ROI

    Example:
```python
        scorer = EffectivenessScorer()
        score = scorer.calculate_score(
            avg_engagement=150.0,
            reach=5000,
            competition_level=0.4,
            momentum=0.8,
            roi=3.5
        )

        print(f"Effectiveness: {score:.1f}/100")
```
    """

    def __init__(self) -> None:
        """Initialize effectiveness scorer."""
        # Weights for each factor
        self.weights = {
            "engagement": 0.30,
            "reach": 0.25,
            "competition": 0.20,
            "momentum": 0.15,
            "roi": 0.10,
        }

    def calculate_score(
        self,
        avg_engagement: float,
        reach: int,
        competition_level: float,
        momentum: float,
        roi: float,
    ) -> float:
        """
        Calculate effectiveness score (0-100).

        Args:
            avg_engagement: Average engagement
            reach: Reach count
            competition_level: Competition (0-1, lower is better)
            momentum: Trend momentum (0-1)
            roi: ROI per character

        Returns:
            Effectiveness score (0-100)
        """
        # Normalize each factor to 0-1
        engagement_score = self._normalize_engagement(avg_engagement)
        reach_score = self._normalize_reach(reach)
        competition_score = 1.0 - competition_level  # Invert (lower competition = better)
        momentum_score = momentum
        roi_score = self._normalize_roi(roi)

        # Weighted sum
        effectiveness = (
            engagement_score * self.weights["engagement"]
            + reach_score * self.weights["reach"]
            + competition_score * self.weights["competition"]
            + momentum_score * self.weights["momentum"]
            + roi_score * self.weights["roi"]
        ) * 100

        return max(0.0, min(100.0, effectiveness))

    def calculate_batch(
        self,
        hashtags_data: List[Dict[str, float]],
    ) -> List[float]:
        """
        Calculate scores for multiple hashtags.

        Args:
            hashtags_data: List of hashtag data dicts

        Returns:
            List of effectiveness scores
        """
        scores = []
        for data in hashtags_data:
            score = self.calculate_score(
                avg_engagement=data.get("avg_engagement", 0.0),
                reach=int(data.get("reach", 0)),
                competition_level=data.get("competition_level", 0.5),
                momentum=data.get("momentum", 0.5),
                roi=data.get("roi", 1.0),
            )
            scores.append(score)
        return scores

    def _normalize_engagement(self, engagement: float) -> float:
        """Normalize engagement to 0-1."""
        # Assume max engagement of 500
        return min(1.0, engagement / 500.0)

    def _normalize_reach(self, reach: int) -> float:
        """Normalize reach to 0-1."""
        # Assume max reach of 50000
        return min(1.0, reach / 50000.0)

    def _normalize_roi(self, roi: float) -> float:
        """Normalize ROI to 0-1."""
        # Assume max ROI of 10
        return min(1.0, roi / 10.0)