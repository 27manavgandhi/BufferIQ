"""
Hashtag strategy generator.

Generates platform-specific hashtag strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS


@dataclass
class HashtagStrategy:
    """Hashtag strategy recommendation."""

    platform: str
    content_topic: str

    # Recommendations
    recommended_count: int
    recommended_hashtags: List[str]

    # Mix breakdown
    broad_hashtags: List[str] = field(default_factory=list)
    niche_hashtags: List[str] = field(default_factory=list)
    branded_hashtags: List[str] = field(default_factory=list)

    # Placement
    placement: str = "end"  # "beginning", "end", "first_comment"
    formatting: str = "spaced"  # How to format

    # Rotation
    rotation_schedule: Optional[Dict[str, List[str]]] = None

    # Expected performance
    predicted_engagement: float = 0.0
    predicted_reach: int = 0
    confidence: float = 0.0


class HashtagStrategyGenerator:
    """
    Generate optimal hashtag strategies by platform.

    Considers platform best practices, content topic,
    and performance data.

    Example:
```python
        generator = HashtagStrategyGenerator()
        strategy = generator.generate(
            platform="linkedin",
            content_topic="artificial intelligence",
            user_profile=user_voice_profile
        )

        print(f"Strategy for {strategy.platform}")
        print(f"Recommended count: {strategy.recommended_count}")
        print(f"\nHashtags:")
        for ht in strategy.recommended_hashtags:
            print(f"  #{ht}")

        print(f"\nMix:")
        print(f"  Broad: {strategy.broad_hashtags}")
        print(f"  Niche: {strategy.niche_hashtags}")
        print(f"  Branded: {strategy.branded_hashtags}")

        print(f"\nPlacement: {strategy.placement}")
        print(f"Predicted engagement: {strategy.predicted_engagement:.1f}")
```
    """

    def __init__(self) -> None:
        """Initialize strategy generator."""
        # Platform-specific rules
        self.platform_rules = {
            "linkedin": {
                "min_hashtags": 3,
                "max_hashtags": 5,
                "optimal_hashtags": 5,
                "placement": "end",
                "mix": {"broad": 0.4, "niche": 0.4, "branded": 0.2},
            },
            "twitter": {
                "min_hashtags": 1,
                "max_hashtags": 2,
                "optimal_hashtags": 2,
                "placement": "end",
                "mix": {"broad": 0.5, "niche": 0.5, "branded": 0.0},
            },
            "bluesky": {
                "min_hashtags": 1,
                "max_hashtags": 3,
                "optimal_hashtags": 2,
                "placement": "end",
                "mix": {"broad": 0.5, "niche": 0.5, "branded": 0.0},
            },
        }

    def generate(
        self,
        platform: str,
        content_topic: str,
        user_profile: Optional[Any] = None,
        target_audience: Optional[str] = None,
    ) -> HashtagStrategy:
        """
        Generate hashtag strategy.

        Args:
            platform: Target platform
            content_topic: Content topic/theme
            user_profile: Optional user profile
            target_audience: Optional audience type

        Returns:
            Hashtag strategy

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Platform-specific generation
        if platform == "linkedin":
            return self._generate_linkedin_strategy(
                content_topic, user_profile, target_audience
            )
        elif platform == "twitter":
            return self._generate_twitter_strategy(
                content_topic, user_profile, target_audience
            )
        elif platform == "bluesky":
            return self._generate_bluesky_strategy(
                content_topic, user_profile, target_audience
            )

        raise ValueError(f"No strategy generator for platform: {platform}")

    def _generate_linkedin_strategy(
        self,
        content_topic: str,
        user_profile: Optional[Any],
        target_audience: Optional[str],
    ) -> HashtagStrategy:
        """Generate LinkedIn-specific strategy (3-5 hashtags)."""
        rules = self.platform_rules["linkedin"]

        # Extract keywords from topic
        keywords = self._extract_keywords(content_topic)

        # Build hashtag mix
        broad = self._get_broad_hashtags(keywords, count=2)
        niche = self._get_niche_hashtags(keywords, count=2)
        branded = self._get_branded_hashtags(user_profile, count=1)

        # Combine
        all_hashtags = broad + niche + branded
        recommended_count = min(len(all_hashtags), rules["optimal_hashtags"])

        # Predict performance
        predicted_engagement = self._predict_engagement(
            platform="linkedin",
            hashtag_count=recommended_count,
            topic=content_topic,
        )

        return HashtagStrategy(
            platform="linkedin",
            content_topic=content_topic,
            recommended_count=recommended_count,
            recommended_hashtags=all_hashtags[:recommended_count],
            broad_hashtags=broad,
            niche_hashtags=niche,
            branded_hashtags=branded,
            placement=rules["placement"],
            formatting="spaced",
            predicted_engagement=predicted_engagement,
            predicted_reach=int(predicted_engagement * 8),  # Rough estimate
            confidence=0.85,
        )

    def _generate_twitter_strategy(
        self,
        content_topic: str,
        user_profile: Optional[Any],
        target_audience: Optional[str],
    ) -> HashtagStrategy:
        """Generate Twitter-specific strategy (1-2 hashtags)."""
        rules = self.platform_rules["twitter"]

        keywords = self._extract_keywords(content_topic)

        # Twitter: fewer, more impactful hashtags
        broad = self._get_broad_hashtags(keywords, count=1)
        niche = self._get_niche_hashtags(keywords, count=1)

        all_hashtags = broad + niche
        recommended_count = min(len(all_hashtags), rules["optimal_hashtags"])

        predicted_engagement = self._predict_engagement(
            platform="twitter",
            hashtag_count=recommended_count,
            topic=content_topic,
        )

        return HashtagStrategy(
            platform="twitter",
            content_topic=content_topic,
            recommended_count=recommended_count,
            recommended_hashtags=all_hashtags[:recommended_count],
            broad_hashtags=broad,
            niche_hashtags=niche,
            branded_hashtags=[],
            placement=rules["placement"],
            formatting="spaced",
            predicted_engagement=predicted_engagement,
            predicted_reach=int(predicted_engagement * 12),
            confidence=0.80,
        )

    def _generate_bluesky_strategy(
        self,
        content_topic: str,
        user_profile: Optional[Any],
        target_audience: Optional[str],
    ) -> HashtagStrategy:
        """Generate Bluesky-specific strategy."""
        rules = self.platform_rules["bluesky"]

        keywords = self._extract_keywords(content_topic)

        broad = self._get_broad_hashtags(keywords, count=1)
        niche = self._get_niche_hashtags(keywords, count=1)

        all_hashtags = broad + niche
        recommended_count = min(len(all_hashtags), rules["optimal_hashtags"])

        predicted_engagement = self._predict_engagement(
            platform="bluesky",
            hashtag_count=recommended_count,
            topic=content_topic,
        )

        return HashtagStrategy(
            platform="bluesky",
            content_topic=content_topic,
            recommended_count=recommended_count,
            recommended_hashtags=all_hashtags[:recommended_count],
            broad_hashtags=broad,
            niche_hashtags=niche,
            branded_hashtags=[],
            placement=rules["placement"],
            formatting="spaced",
            predicted_engagement=predicted_engagement,
            predicted_reach=int(predicted_engagement * 10),
            confidence=0.75,
        )

    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract keywords from topic."""
        # Simple keyword extraction
        words = topic.lower().split()
        # Remove common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        keywords = [w for w in words if w not in stopwords]
        return keywords

    def _get_broad_hashtags(self, keywords: List[str], count: int) -> List[str]:
        """Get broad, high-volume hashtags."""
        # Map keywords to broad hashtags
        broad_map = {
            "ai": "ai",
            "artificial": "ai",
            "intelligence": "ai",
            "machine": "machinelearning",
            "learning": "machinelearning",
            "marketing": "marketing",
            "digital": "digitalmarketing",
            "business": "business",
            "tech": "technology",
            "innovation": "innovation",
        }

        broad = []
        for keyword in keywords:
            if keyword in broad_map:
                hashtag = broad_map[keyword]
                if hashtag not in broad:
                    broad.append(hashtag)

        return broad[:count]

    def _get_niche_hashtags(self, keywords: List[str], count: int) -> List[str]:
        """Get niche, targeted hashtags."""
        # Create niche variants
        niche = []
        for keyword in keywords[:count]:
            niche.append(f"{keyword}tips")

        return niche[:count]

    def _get_branded_hashtags(
        self, user_profile: Optional[Any], count: int
    ) -> List[str]:
        """Get branded hashtags."""
        if user_profile and hasattr(user_profile, "brand_name"):
            return [user_profile.brand_name.lower().replace(" ", "")][:count]
        return []

    def _predict_engagement(
        self, platform: str, hashtag_count: int, topic: str
    ) -> float:
        """Predict engagement based on strategy."""
        # Base engagement
        base = 100.0

        # Platform multiplier
        platform_mult = {"linkedin": 1.2, "twitter": 1.0, "bluesky": 0.9}
        base *= platform_mult.get(platform, 1.0)

        # Hashtag count factor
        if hashtag_count > 0:
            base *= 1.0 + (hashtag_count * 0.1)

        # Topic popularity (mock)
        if "ai" in topic.lower() or "tech" in topic.lower():
            base *= 1.3

        return base