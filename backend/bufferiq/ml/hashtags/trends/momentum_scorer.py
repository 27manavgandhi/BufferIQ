"""
Momentum scorer for trending hashtags.

Calculates momentum based on volume and velocity.
"""

from typing import List, Tuple
import numpy as np


class MomentumScorer:
    """
    Calculate momentum scores for hashtags.

    Example:
```python
        scorer = MomentumScorer()
        momentum = scorer.calculate(
            current_volume=1500,
            previous_volume=1000,
            velocity=0.5
        )

        print(f"Momentum: {momentum:.1f}/100")
```
    """

    def calculate(
        self,
        current_volume: int,
        previous_volume: int,
        velocity: float,
        volume_weight: float = 0.6,
        velocity_weight: float = 0.4,
    ) -> float:
        """
        Calculate momentum score (0-100).

        Args:
            current_volume: Current usage volume
            previous_volume: Previous period volume
            velocity: Growth velocity
            volume_weight: Weight for volume component
            velocity_weight: Weight for velocity component

        Returns:
            Momentum score (0-100)
        """
        # Calculate growth rate
        growth = (
            (current_volume - previous_volume) / previous_volume
            if previous_volume > 0
            else 0.0
        )

        # Weighted combination
        momentum = (growth * volume_weight + velocity * velocity_weight) * 100

        # Clamp to valid range
        return max(0.0, min(100.0, momentum))

    def calculate_batch(
        self, volume_pairs: List[Tuple[int, int]], velocities: List[float]
    ) -> List[float]:
        """
        Calculate momentum for multiple hashtags.

        Args:
            volume_pairs: List of (current, previous) volume tuples
            velocities: List of velocities

        Returns:
            List of momentum scores
        """
        scores = []
        for (current, previous), velocity in zip(volume_pairs, velocities):
            score = self.calculate(current, previous, velocity)
            scores.append(score)
        return scores