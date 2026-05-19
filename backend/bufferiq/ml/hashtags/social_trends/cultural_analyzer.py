"""
Cultural moment analyzer.

Identifies cultural moments and events.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class CulturalMoment:
    """Detected cultural moment."""

    theme: str
    hashtags: List[str]
    detected_at: datetime
    impact_score: float  # 0-100
    description: str


class CulturalAnalyzer:
    """
    Analyze cultural moments and events.

    Example:
```python
        analyzer = CulturalAnalyzer()

        moment = analyzer.detect_cultural_moment(
            hashtags=["ai", "chatgpt", "openai"],
            volumes=[5000, 4500, 4000],
            context="technology"
        )

        if moment:
            print(f"Cultural moment: {moment.theme}")
            print(f"  Impact: {moment.impact_score:.1f}")
```
    """

    def detect_cultural_moment(
        self,
        hashtags: List[str],
        volumes: List[int],
        context: str | None = None,
    ) -> CulturalMoment | None:
        """
        Detect cultural moment from hashtag activity.

        Args:
            hashtags: Related hashtags
            volumes: Corresponding volumes
            context: Optional context/category

        Returns:
            Cultural moment if detected
        """
        if len(hashtags) != len(volumes):
            return None

        # Calculate total impact
        total_volume = sum(volumes)
        avg_volume = total_volume / len(volumes) if volumes else 0

        # Check if significant enough
        if total_volume < 1000:  # Threshold
            return None

        # Determine theme
        theme = self._determine_theme(hashtags, context)

        # Calculate impact score
        impact_score = min(100.0, (total_volume / 1000.0) * 10)

        # Generate description
        description = f"Cultural moment around {theme} with {len(hashtags)} related hashtags"

        return CulturalMoment(
            theme=theme,
            hashtags=hashtags,
            detected_at=datetime.now(),
            impact_score=impact_score,
            description=description,
        )

    def _determine_theme(self, hashtags: List[str], context: str | None) -> str:
        """Determine theme from hashtags."""
        if context:
            return context

        # Simple theme detection based on common patterns
        tech_keywords = {"ai", "ml", "tech", "digital", "innovation"}
        business_keywords = {"business", "startup", "entrepreneur", "marketing"}

        # Count keyword matches
        tech_count = sum(1 for ht in hashtags if any(kw in ht for kw in tech_keywords))
        biz_count = sum(
            1 for ht in hashtags if any(kw in ht for kw in business_keywords)
        )

        if tech_count > biz_count:
            return "technology"
        elif biz_count > tech_count:
            return "business"
        else:
            return "general"