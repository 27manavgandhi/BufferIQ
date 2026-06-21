"""Content preference modeling."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import AudienceDataPoint


class ContentPreferenceModeler:
    """Model content type preferences."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize content preference modeler."""
        self.config = config or {}

    def model(
        self, audience_members: List[AudienceDataPoint]
    ) -> Dict[str, Any]:
        """
        Model content preferences.

        Args:
            audience_members: List of audience members

        Returns:
            Dictionary of content preferences
        """
        if not audience_members:
            return self._empty_preferences()

        # Calculate format preferences
        format_preferences = self._calculate_format_preferences(audience_members)

        # Calculate content length preferences
        length_preference = self._determine_length_preference(audience_members)

        # Calculate tone preferences
        tone_preference = self._determine_tone_preference(audience_members)

        return {
            "formats": format_preferences,
            "length": length_preference,
            "tone": tone_preference,
        }

    def _calculate_format_preferences(
        self, audience_members: List[AudienceDataPoint]
    ) -> Dict[str, float]:
        """Calculate preferences by content format."""
        formats = ["text", "image", "video", "link"]
        preferences: Dict[str, float] = {}

        for fmt in formats:
            count = sum(
                1 for m in audience_members if fmt in m.content_types_engaged
            )
            preferences[fmt] = count / len(audience_members) if audience_members else 0.0

        return preferences

    def _determine_length_preference(
        self, audience_members: List[AudienceDataPoint]
    ) -> str:
        """Determine content length preference."""
        # Based on engagement rate and topic diversity
        avg_engagement = sum(
            m.avg_engagement_rate for m in audience_members
        ) / len(audience_members)

        if avg_engagement > 0.7:
            return "short"  # High engagement with short content
        elif avg_engagement > 0.4:
            return "medium"
        else:
            return "long"  # Lower engagement may prefer comprehensive content

    def _determine_tone_preference(
        self, audience_members: List[AudienceDataPoint]
    ) -> str:
        """Determine tone preference."""
        verified_ratio = sum(1 for m in audience_members if m.verified) / len(
            audience_members
        )

        if verified_ratio > 0.5:
            return "professional"
        else:
            return "casual"

    def _empty_preferences(self) -> Dict[str, Any]:
        """Return empty preferences."""
        return {
            "formats": {"text": 0.25, "image": 0.25, "video": 0.25, "link": 0.25},
            "length": "medium",
            "tone": "casual",
        }