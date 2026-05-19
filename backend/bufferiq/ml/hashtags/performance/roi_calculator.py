"""
ROI calculator for hashtags.

Calculates return on investment per character.
"""

from typing import Dict


class ROICalculator:
    """
    Calculate ROI for hashtags.

    Measures engagement gain per character used.

    Example:
```python
        calculator = ROICalculator()
        roi = calculator.calculate(
            avg_engagement_with=150.0,
            avg_engagement_without=120.0,
            hashtag="artificialintelligence"
        )

        print(f"ROI: {roi:.2f} engagement per character")
```
    """

    def calculate(
        self,
        avg_engagement_with: float,
        avg_engagement_without: float,
        hashtag: str,
    ) -> float:
        """
        Calculate ROI per character.

        Args:
            avg_engagement_with: Average engagement with hashtag
            avg_engagement_without: Average engagement without
            hashtag: The hashtag (without #)

        Returns:
            Engagement gain per character
        """
        # Character cost includes # symbol
        char_cost = len(hashtag) + 1

        # Engagement gain
        gain = avg_engagement_with - avg_engagement_without

        # ROI per character
        roi = gain / char_cost if char_cost > 0 else 0.0

        return roi

    def calculate_efficiency_score(
        self, roi: float, hashtag_length: int
    ) -> float:
        """
        Calculate efficiency score (0-100).

        Shorter hashtags with high ROI score higher.

        Args:
            roi: ROI per character
            hashtag_length: Length of hashtag

        Returns:
            Efficiency score (0-100)
        """
        # Normalize ROI (assume max ROI of 10)
        normalized_roi = min(1.0, roi / 10.0)

        # Penalty for length (prefer shorter)
        length_factor = 1.0 - (min(hashtag_length, 30) / 30.0) * 0.3

        # Combine
        score = normalized_roi * length_factor * 100

        return max(0.0, min(100.0, score))