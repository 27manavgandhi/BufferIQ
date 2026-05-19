"""
Twitter hashtag optimizer.

Platform-specific optimization for Twitter (1-2 hashtags).
"""

from typing import Dict, List


class TwitterOptimizer:
    """
    Optimize hashtags for Twitter.

    Twitter best practices:
    - 1-2 hashtags optimal
    - Concise
    - Trending-aware
    - High impact

    Example:
```python
        optimizer = TwitterOptimizer()

        optimized = optimizer.optimize(
            content="AI is transforming business",
            available_hashtags=["ai", "tech", "ml", "business"]
        )

        print(f"Optimized for Twitter: {optimized}")
```
    """

    def __init__(self) -> None:
        """Initialize Twitter optimizer."""
        self.min_hashtags = 1
        self.max_hashtags = 2
        self.optimal_count = 2

    def optimize(
        self,
        content: str,
        available_hashtags: List[str],
        effectiveness_scores: Dict[str, float] | None = None,
    ) -> List[str]:
        """
        Optimize hashtags for Twitter.

        Args:
            content: Tweet content
            available_hashtags: Available hashtags
            effectiveness_scores: Optional effectiveness scores

        Returns:
            Optimized hashtag list (1-2 hashtags)
        """
        if effectiveness_scores is None:
            effectiveness_scores = {}

        # Sort by effectiveness
        sorted_hashtags = sorted(
            available_hashtags,
            key=lambda h: effectiveness_scores.get(h, 50.0),
            reverse=True,
        )

        # Select top 2 (or 1)
        selected = sorted_hashtags[: self.optimal_count]

        return selected

    def validate(self, hashtags: List[str]) -> Dict[str, bool | str]:
        """
        Validate hashtags for Twitter.

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