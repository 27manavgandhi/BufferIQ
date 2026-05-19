"""
Related hashtag finder using co-occurrence analysis.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


class RelatedHashtagFinder:
    """
    Find related hashtags through co-occurrence analysis.

    Example:
```python
        finder = RelatedHashtagFinder()
        finder.add_post(["ai", "machinelearning", "tech"])
        finder.add_post(["ai", "innovation", "tech"])

        related = finder.find_related("ai", min_score=0.3)
        for hashtag, score in related:
            print(f"#{hashtag}: {score:.2f}")
```
    """

    def __init__(self) -> None:
        """Initialize related finder."""
        self.cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.hashtag_counts: Dict[str, int] = defaultdict(int)

    def add_post(self, hashtags: List[str]) -> None:
        """
        Add post hashtags to co-occurrence matrix.

        Args:
            hashtags: List of hashtags in post
        """
        # Update individual counts
        for hashtag in hashtags:
            self.hashtag_counts[hashtag] += 1

        # Update co-occurrence
        unique_hashtags = list(set(hashtags))
        for i, tag1 in enumerate(unique_hashtags):
            for tag2 in unique_hashtags[i + 1 :]:
                pair = tuple(sorted([tag1, tag2]))
                self.cooccurrence[pair] += 1

    def find_related(
        self, hashtag: str, min_score: float = 0.3, limit: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Find related hashtags.

        Args:
            hashtag: Seed hashtag
            min_score: Minimum similarity score
            limit: Maximum results

        Returns:
            List of (hashtag, similarity_score) tuples
        """
        if hashtag not in self.hashtag_counts:
            return []

        related: List[Tuple[str, float]] = []

        # Calculate similarity with each hashtag
        for other_hashtag in self.hashtag_counts:
            if other_hashtag == hashtag:
                continue

            score = self._calculate_similarity(hashtag, other_hashtag)
            if score >= min_score:
                related.append((other_hashtag, score))

        # Sort by score
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:limit]

    def _calculate_similarity(self, hashtag1: str, hashtag2: str) -> float:
        """
        Calculate Jaccard similarity between hashtags.

        Args:
            hashtag1: First hashtag
            hashtag2: Second hashtag

        Returns:
            Similarity score (0-1)
        """
        pair = tuple(sorted([hashtag1, hashtag2]))
        cooccur_count = self.cooccurrence.get(pair, 0)

        # Jaccard similarity: intersection / union
        count1 = self.hashtag_counts[hashtag1]
        count2 = self.hashtag_counts[hashtag2]

        union = count1 + count2 - cooccur_count

        if union == 0:
            return 0.0

        similarity = cooccur_count / union
        return similarity