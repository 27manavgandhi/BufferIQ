"""
Diversity optimizer for hashtag sets.

Ensures hashtag sets are diverse and balanced.
"""

from typing import List
import numpy as np


class DiversityOptimizer:
    """
    Optimize hashtag set diversity.

    Example:
```python
        optimizer = DiversityOptimizer()

        diversity = optimizer.calculate_diversity(
            hashtags=["ai", "machinelearning", "tech"]
        )

        print(f"Diversity: {diversity:.1f}/100")
```
    """

    def calculate_diversity(self, hashtags: List[str]) -> float:
        """
        Calculate diversity score (0-100).

        Considers:
        - Character uniqueness
        - Length variance
        - Semantic diversity (approximated)

        Args:
            hashtags: List of hashtags

        Returns:
            Diversity score (0-100)
        """
        if not hashtags:
            return 0.0

        if len(hashtags) == 1:
            return 50.0  # Neutral for single hashtag

        # Character uniqueness
        all_chars = "".join(hashtags)
        unique_chars = len(set(all_chars))
        total_chars = len(all_chars)
        char_diversity = unique_chars / total_chars if total_chars > 0 else 0

        # Length variance
        lengths = [len(ht) for ht in hashtags]
        if len(lengths) > 1:
            length_std = np.std(lengths)
            length_mean = np.mean(lengths)
            length_variance = length_std / length_mean if length_mean > 0 else 0
        else:
            length_variance = 0

        # Substring overlap (lower = more diverse)
        overlap_score = 1.0 - self._calculate_overlap(hashtags)

        # Weighted combination
        diversity = (
            char_diversity * 0.4 + length_variance * 0.2 + overlap_score * 0.4
        ) * 100

        return max(0.0, min(100.0, diversity))

    def optimize_for_diversity(
        self,
        hashtags: List[str],
        target_count: int,
        min_diversity: float = 60.0,
    ) -> List[str]:
        """
        Select subset that maximizes diversity.

        Args:
            hashtags: Available hashtags
            target_count: Number to select
            min_diversity: Minimum diversity threshold

        Returns:
            Optimized subset
        """
        if target_count >= len(hashtags):
            return hashtags

        # Start with most different hashtags
        selected: List[str] = [hashtags[0]]

        remaining = hashtags[1:]

        while len(selected) < target_count and remaining:
            # Find hashtag most different from selected
            best_candidate = None
            best_diversity = -1.0

            for candidate in remaining:
                test_set = selected + [candidate]
                diversity = self.calculate_diversity(test_set)

                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate and best_diversity >= min_diversity:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # Can't meet diversity threshold
                break

        return selected

    def _calculate_overlap(self, hashtags: List[str]) -> float:
        """
        Calculate average substring overlap.

        Args:
            hashtags: List of hashtags

        Returns:
            Overlap ratio (0-1)
        """
        if len(hashtags) < 2:
            return 0.0

        total_overlap = 0.0
        comparisons = 0

        for i, ht1 in enumerate(hashtags):
            for ht2 in hashtags[i + 1 :]:
                # Calculate character overlap
                chars1 = set(ht1)
                chars2 = set(ht2)
                overlap = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0
                total_overlap += overlap
                comparisons += 1

        avg_overlap = total_overlap / comparisons if comparisons > 0 else 0.0

        return avg_overlap