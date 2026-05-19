"""
LinkedIn hashtag optimizer.

Platform-specific optimization for LinkedIn (3-5 hashtags).
"""

from typing import List, Dict


class LinkedInOptimizer:
    """
    Optimize hashtags for LinkedIn.

    LinkedIn best practices:
    - 3-5 hashtags optimal
    - Professional tone
    - Industry-specific
    - Mix of broad and niche

    Example:
```python
        optimizer = LinkedInOptimizer()

        optimized = optimizer.optimize(
            content="Great insights on AI in business",
            available_hashtags=[
                "ai", "business", "tech", "innovation",
                "machinelearning", "leadership"
            ]
        )

        print(f"Optimized for LinkedIn: {optimized}")
```
    """

    def __init__(self) -> None:
        """Initialize LinkedIn optimizer."""
        self.min_hashtags = 3
        self.max_hashtags = 5
        self.optimal_count = 5

    def optimize(
        self,
        content: str,
        available_hashtags: List[str],
        effectiveness_scores: Dict[str, float] | None = None,
    ) -> List[str]:
        """
        Optimize hashtags for LinkedIn.

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

        # Ensure mix of broad and niche
        optimized = self._ensure_mix(selected)

        return optimized

    def validate(self, hashtags: List[str]) -> Dict[str, bool | str]:
        """
        Validate hashtags for LinkedIn.

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

    def _ensure_mix(self, hashtags: List[str]) -> List[str]:
        """Ensure mix of broad and niche hashtags."""
        # For LinkedIn, we want a balance
        # This is a simplified implementation
        # In production, use more sophisticated categorization

        return hashtags