"""
Hashtag combination optimizer.

Finds optimal combinations of hashtags.
"""

from typing import List, Tuple
from itertools import combinations
import numpy as np


class CombinationOptimizer:
    """
    Optimize hashtag combinations.

    Finds combinations that maximize engagement while
    maintaining diversity.

    Example:
```python
        optimizer = CombinationOptimizer()

        best_combo = optimizer.find_best_combination(
            hashtags=["ai", "ml", "tech", "innovation", "data"],
            count=5,
            effectiveness_scores={
                "ai": 85, "ml": 80, "tech": 75,
                "innovation": 70, "data": 78
            }
        )

        print(f"Best combination: {best_combo}")
```
    """

    def __init__(self) -> None:
        """Initialize combination optimizer."""
        pass

    def find_best_combination(
        self,
        hashtags: List[str],
        count: int,
        effectiveness_scores: dict[str, float],
        diversity_weight: float = 0.3,
    ) -> List[str]:
        """
        Find best combination of hashtags.

        Args:
            hashtags: Available hashtags
            count: Number to select
            effectiveness_scores: Effectiveness score for each
            diversity_weight: Weight for diversity (0-1)

        Returns:
            Best combination
        """
        if count >= len(hashtags):
            return hashtags

        # Generate all possible combinations
        all_combinations = list(combinations(hashtags, count))

        best_score = -1.0
        best_combo: List[str] = []

        for combo in all_combinations:
            # Calculate combined score
            effectiveness = sum(
                effectiveness_scores.get(ht, 50.0) for ht in combo
            ) / count

            diversity = self._calculate_diversity(list(combo))

            # Weighted combination
            total_score = (
                effectiveness * (1 - diversity_weight) + diversity * diversity_weight
            )

            if total_score > best_score:
                best_score = total_score
                best_combo = list(combo)

        return best_combo

    def _calculate_diversity(self, hashtags: List[str]) -> float:
        """
        Calculate diversity score (0-100).

        Args:
            hashtags: List of hashtags

        Returns:
            Diversity score
        """
        if len(hashtags) <= 1:
            return 0.0

        # Calculate character overlap
        all_chars = "".join(hashtags)
        unique_chars = len(set(all_chars))
        total_chars = len(all_chars)

        char_diversity = unique_chars / total_chars if total_chars > 0 else 0

        # Calculate length variance
        lengths = [len(ht) for ht in hashtags]
        length_variance = np.std(lengths) / np.mean(lengths) if lengths else 0

        # Combine
        diversity = (char_diversity * 0.7 + length_variance * 0.3) * 100

        return min(100.0, diversity)