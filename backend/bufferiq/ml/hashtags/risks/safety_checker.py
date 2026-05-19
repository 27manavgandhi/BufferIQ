"""
Brand safety checker for hashtags.

Ensures hashtags align with brand safety guidelines.
"""

from typing import List, Set


class SafetyChecker:
    """
    Check hashtag brand safety.

    Example:
```python
        checker = SafetyChecker()

        is_safe = checker.is_brand_safe(
            hashtag="ai",
            brand_guidelines=["professional", "technology"]
        )

        print(f"Brand safe: {is_safe}")
```
    """

    def __init__(self) -> None:
        """Initialize safety checker."""
        # Categories that need extra caution
        self.sensitive_categories = {
            "politics",
            "religion",
            "health",
            "finance",
            "legal",
        }

        # Always unsafe
        self.unsafe_keywords = {
            "hate",
            "violence",
            "discrimination",
        }

    def is_brand_safe(
        self,
        hashtag: str,
        brand_guidelines: List[str] | None = None,
    ) -> bool:
        """
        Check if hashtag is brand safe.

        Args:
            hashtag: Hashtag to check
            brand_guidelines: Optional brand safety guidelines

        Returns:
            True if safe
        """
        hashtag_lower = hashtag.lower().lstrip("#")

        # Check for unsafe keywords
        if any(keyword in hashtag_lower for keyword in self.unsafe_keywords):
            return False

        # Check sensitive categories if guidelines provided
        if brand_guidelines:
            # Ensure hashtag aligns with guidelines
            matches_guidelines = any(
                guideline.lower() in hashtag_lower for guideline in brand_guidelines
            )

            # If it doesn't match and is sensitive, mark as unsafe
            if not matches_guidelines and self._is_sensitive(hashtag_lower):
                return False

        return True

    def get_safety_score(
        self,
        hashtag: str,
        brand_guidelines: List[str] | None = None,
    ) -> float:
        """
        Get safety score (0-100).

        Args:
            hashtag: Hashtag to check
            brand_guidelines: Optional guidelines

        Returns:
            Safety score (higher is safer)
        """
        score = 100.0

        hashtag_lower = hashtag.lower().lstrip("#")

        # Deduct for unsafe keywords
        for keyword in self.unsafe_keywords:
            if keyword in hashtag_lower:
                score -= 50.0

        # Deduct for sensitive without guidelines
        if self._is_sensitive(hashtag_lower) and not brand_guidelines:
            score -= 20.0

        # Bonus for alignment with guidelines
        if brand_guidelines:
            matches = sum(
                1 for guideline in brand_guidelines if guideline.lower() in hashtag_lower
            )
            score += matches * 10.0

        return max(0.0, min(100.0, score))

    def _is_sensitive(self, hashtag: str) -> bool:
        """Check if hashtag is in sensitive category."""
        return any(category in hashtag for category in self.sensitive_categories)