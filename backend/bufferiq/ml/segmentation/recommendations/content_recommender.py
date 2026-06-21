"""Content recommendations for segments."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import PersonaProfile


class ContentRecommender:
    """Generate content recommendations for personas."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize content recommender."""
        self.config = config or {}

    def recommend(self, persona: PersonaProfile) -> Dict[str, Any]:
        """
        Generate content recommendations.

        Args:
            persona: Persona profile

        Returns:
            Content recommendations
        """
        topics = self._recommend_topics(persona)
        formats = self._recommend_formats(persona)
        hooks = self._generate_hooks(persona, topics)

        return {
            "topics": topics,
            "formats": formats,
            "sample_hooks": hooks,
        }

    def _recommend_topics(self, persona: PersonaProfile) -> List[str]:
        """Recommend topics for persona."""
        return persona.primary_topics + persona.secondary_topics

    def _recommend_formats(self, persona: PersonaProfile) -> List[str]:
        """Recommend content formats."""
        preferences = persona.content_type_preferences
        if not preferences:
            return ["text", "image", "video"]

        # Sort by preference score
        sorted_formats = sorted(
            preferences.items(), key=lambda x: x[1], reverse=True
        )
        return [fmt for fmt, _ in sorted_formats[:3]]

    def _generate_hooks(
        self, persona: PersonaProfile, topics: List[str]
    ) -> List[str]:
        """Generate sample hooks."""
        if not topics:
            topics = ["your interests"]

        hooks = []
        for topic in topics[:2]:
            hook = f"Discover the latest in {topic}"
            hooks.append(hook)

        return hooks[:3]