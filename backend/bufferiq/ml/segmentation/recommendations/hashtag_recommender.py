"""Hashtag recommendations for segments."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import PersonaProfile


class HashtagRecommender:
    """Generate hashtag recommendations."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize hashtag recommender."""
        self.config = config or {}
        self.max_hashtags = self.config.get("max_hashtags", 5)

    def recommend(self, persona: PersonaProfile) -> Dict[str, Any]:
        """
        Generate hashtag recommendations.

        Args:
            persona: Persona profile

        Returns:
            Hashtag recommendations
        """
        hashtags = self._generate_hashtags(persona)
        count = len(hashtags)

        return {
            "recommended_hashtags": hashtags,
            "hashtag_count": count,
            "max_hashtags": self.max_hashtags,
        }

    def _generate_hashtags(self, persona: PersonaProfile) -> List[str]:
        """Generate hashtags from persona topics."""
        hashtags = []

        # Generate from primary topics
        for topic in persona.primary_topics[:3]:
            hashtag = f"#{topic.replace(' ', '').lower()}"
            hashtags.append(hashtag)

        # Add platform-specific hashtags
        platform_hashtags = self._get_platform_hashtags(persona.platform)
        hashtags.extend(platform_hashtags)

        return hashtags[: self.max_hashtags]

    def _get_platform_hashtags(self, platform: str) -> List[str]:
        """Get platform-specific hashtags."""
        platform_tags = {
            "linkedin": ["#professional", "#career", "#business"],
            "twitter": ["#trending", "#follow", "#engagement"],
            "bluesky": ["#community", "#openplatform", "#decentralized"],
        }

        return platform_tags.get(platform, [])