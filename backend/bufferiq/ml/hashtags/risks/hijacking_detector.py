"""
Hashtag hijacking detector.

Detects when hashtags have been hijacked for different purposes.
"""

from typing import Dict, List
from datetime import datetime, timedelta


class HijackingDetector:
    """
    Detect hashtag hijacking.

    Identifies when hashtags are being used for unintended purposes.

    Example:
```python
        detector = HijackingDetector()

        is_hijacked = detector.check_hijacking(
            hashtag="example",
            original_context="technology",
            recent_contexts=["spam", "promotion", "spam"]
        )

        if is_hijacked:
            print("Warning: Hashtag may be hijacked")
```
    """

    def __init__(self, context_shift_threshold: float = 0.7) -> None:
        """
        Initialize hijacking detector.

        Args:
            context_shift_threshold: Threshold for context shift (0-1)
        """
        self.context_shift_threshold = context_shift_threshold

    def check_hijacking(
        self,
        hashtag: str,
        original_context: str,
        recent_contexts: List[str],
    ) -> bool:
        """
        Check if hashtag has been hijacked.

        Args:
            hashtag: Hashtag to check
            original_context: Original intended context
            recent_contexts: Recent usage contexts

        Returns:
            True if likely hijacked
        """
        if not recent_contexts:
            return False

        # Calculate context shift
        different_contexts = sum(
            1 for context in recent_contexts if context != original_context
        )

        shift_ratio = different_contexts / len(recent_contexts)

        return shift_ratio >= self.context_shift_threshold

    def get_hijacking_score(
        self,
        original_context: str,
        recent_contexts: List[str],
    ) -> float:
        """
        Get hijacking probability score (0-100).

        Args:
            original_context: Original context
            recent_contexts: Recent contexts

        Returns:
            Hijacking probability
        """
        if not recent_contexts:
            return 0.0

        # Calculate context shift
        different = sum(1 for ctx in recent_contexts if ctx != original_context)
        shift_ratio = different / len(recent_contexts)

        # Convert to 0-100
        return shift_ratio * 100