"""Style and tone recommendations for segments."""

from typing import Any, Dict

from bufferiq.ml.segmentation.types import PersonaProfile, SUPPORTED_PLATFORMS


class StyleRecommender:
    """Generate style and tone recommendations."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize style recommender."""
        self.config = config or {}

    def recommend(self, persona: PersonaProfile) -> Dict[str, Any]:
        """
        Generate style recommendations.

        Args:
            persona: Persona profile

        Returns:
            Style recommendations
        """
        tone = self._determine_tone(persona)
        vocabulary = self._determine_vocabulary(persona)
        emoji_usage = self._determine_emoji_usage(persona)
        length = persona.recommended_length or "medium"

        return {
            "tone": tone,
            "vocabulary_level": vocabulary,
            "emoji_usage": emoji_usage,
            "content_length": length,
        }

    def _determine_tone(self, persona: PersonaProfile) -> str:
        """Determine recommended tone."""
        if persona.platform == "linkedin":
            return "professional"
        elif persona.platform == "twitter":
            return "conversational"
        else:  # bluesky
            return "friendly"

    def _determine_vocabulary(self, persona: PersonaProfile) -> str:
        """Determine vocabulary level."""
        if persona.verified_ratio > 0.5:
            return "advanced"
        elif persona.avg_engagement_rate > 0.6:
            return "moderate"
        else:
            return "simple"

    def _determine_emoji_usage(self, persona: PersonaProfile) -> str:
        """Determine emoji usage level."""
        if persona.platform == "linkedin":
            return "none"
        elif persona.platform == "twitter":
            return "moderate"
        else:
            return "minimal"