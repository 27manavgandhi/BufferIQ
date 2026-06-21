"""Persona naming and description generation."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.constants import (
    ARCHETYPES_BY_PLATFORM,
    ADJECTIVES_BY_BEHAVIOR,
)


class PersonaNamer:
    """
    Generate evocative persona names and descriptions.

    Names follow the pattern: [Adjective] [Archetype]
    E.g.: "The Engaged Professional", "The Casual Browser"
    Platform-specific naming conventions applied.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize persona namer."""
        self.config = config or {}

    def generate_name(
        self,
        demographics: Dict[str, Any],
        behaviors: Dict[str, Any],
        interests: Dict[str, List[str]],
        platform: str,
    ) -> str:
        """
        Generate persona name.

        Args:
            demographics: Inferred demographic data
            behaviors: Behavioral profile data
            interests: Interest mapping data
            platform: Platform type

        Returns:
            Evocative persona name
        """
        # Select archetype based on platform
        archetypes = ARCHETYPES_BY_PLATFORM.get(platform, ["Professional"])

        # Determine primary behavior category
        behavior_category = self._classify_behavior(behaviors)

        # Select adjective
        adjectives = ADJECTIVES_BY_BEHAVIOR.get(behavior_category, ["Engaged"])
        adjective_index = abs(hash(str(demographics))) % len(adjectives)
        adjective = adjectives[adjective_index]

        # Select archetype based on interests
        archetype = self._select_archetype(archetypes, interests, behaviors)

        return f"The {adjective} {archetype}"

    def generate_description(
        self,
        demographics: Dict[str, Any],
        behaviors: Dict[str, Any],
        interests: Dict[str, List[str]],
        platform: str,
    ) -> str:
        """
        Generate persona description.

        Args:
            demographics: Demographic data
            behaviors: Behavioral data
            interests: Interest data
            platform: Platform type

        Returns:
            Descriptive text for persona
        """
        age_min, age_max = demographics.get("age_range", (25, 55))
        primary_topics = ", ".join(interests.get("primary", ["general"])[:3])
        interaction_type = behaviors.get("primary_interaction", "likes")
        engagement = behaviors.get("avg_engagement", 0.5)

        engagement_level = (
            "highly engaged" if engagement > 0.7 else (
                "moderately engaged" if engagement > 0.4 else "occasionally engaged"
            )
        )

        description = (
            f"A {engagement_level} audience member interested in {primary_topics}. "
            f"Typically aged {age_min}-{age_max}, they prefer {interaction_type} "
            f"as their primary form of engagement."
        )

        return description

    def _classify_behavior(self, behaviors: Dict[str, Any]) -> str:
        """Classify behavior into categories."""
        avg_engagement = behaviors.get("avg_engagement", 0.5)
        primary_interaction = behaviors.get("primary_interaction", "likes")

        # Determine engagement level
        if avg_engagement > 0.7:
            engagement_level = "high_engagement"
        else:
            engagement_level = "low_engagement"

        # Determine content interaction pattern
        if primary_interaction in ["shares", "retweets"]:
            interaction_pattern = "content_creator"
        else:
            interaction_pattern = "content_consumer"

        return engagement_level

    def _select_archetype(
        self,
        archetypes: List[str],
        interests: Dict[str, List[str]],
        behaviors: Dict[str, Any],
    ) -> str:
        """Select appropriate archetype."""
        if not archetypes:
            return "Professional"

        # Select based on interests and behaviors
        primary_topics = interests.get("primary", [])

        # Business/professional topics map to first archetype
        business_keywords = ["business", "finance", "technology", "startup"]
        if any(topic.lower() in business_keywords for topic in primary_topics):
            return archetypes[0]

        # Default to first archetype
        return archetypes[min(len(primary_topics) - 1, len(archetypes) - 1)]