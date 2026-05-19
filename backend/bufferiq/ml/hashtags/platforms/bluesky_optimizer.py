"""
Bluesky hashtag optimizer.

Platform-specific optimization for Bluesky.
"""

from typing import Dict, List


class BlueskyOptimizer:
    """
    Optimize hashtags for Bluesky.

    Bluesky characteristics:
    - Flexible approach (1-3 hashtags)
    - Community-focused
    - Authentic engagement

    Example:
```python
        optimizer = BlueskyOptimizer()

        optimized = optimizer.optimize(
            content="Thoughts on decentralized social media",
            available_hashtags=["web3", "decentralized", "social"]
        )

        print(f"Optimized for Bluesky: {optimized}")
```
    """

    def __init__(self) -> None:
        """Initialize Bluesky optimizer."""
        self.min_hashtags = 1
        self.max_hashtags = 3
        self.optimal_count = 2

    def optimize(
        self,
        content: str,
        available_hashtags: List[str],
        effectiveness_scores: Dict[str, float] | None = None,
    ) -> List[str]:
        """
        Optimize hashtags for Bluesky.

        Args:
            content: Post content
            available_hashtags: Available hashtags
            effectiveness_scores: Optional effectiveness scores

        Returns:
            Optimized hashtag list
        """
        if effectiveness_scores is None:
            effectiveness_scores = {}

        # Sort by effectiveness
        sorted_hashtags = sorted(
            available_hashtags,
            key=lambda h: effectiveness_scores.get(h, 50.0),
            reverse=True,
        )

        # Select optimal count
        selected = sorted_hashtags[: self.optimal_count]

        return selected

    def validate(self, hashtags: List[str]) -> Dict[str, bool | str]:
        """
        Validate hashtags for Bluesky.

        Args:
            hashtags: Hashtags to validate

        Returns:
            Validation result
        """
        is_valid = self.min_hashtags <= len(hashtags) <= self.max_hashtags

        if len(hashtags) < self.min_hashtags:
            message = f"Too few hashtags (min: {self.min_hashtags})"
        elif len(hashtags) > self.max_hashtags:
            message = f"Too many hashtags (max: {self.max_hashtags})"
        else:
            message = "Valid"

        return {
            "is_valid": is_valid,
            "count": len(hashtags),
            "min": self.min_hashtags,
            "max": self.max_hashtags,
            "message": message,
        }