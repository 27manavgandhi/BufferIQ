"""Render persona profile visualizations."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import PersonaProfile


class PersonaRenderer:
    """Render persona profiles for visualization."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize persona renderer."""
        self.config = config or {}

    def render_profile(self, persona: PersonaProfile) -> Dict[str, Any]:
        """
        Render complete persona profile.

        Args:
            persona: Persona to render

        Returns:
            Rendered profile data
        """
        return {
            "name": persona.persona_name,
            "description": persona.persona_description,
            "segment_id": persona.segment_id,
            "platform": persona.platform,
            "size": persona.size,
            "size_percentage": float(persona.size_percentage),
            "demographics": {
                "age_range": persona.estimated_age_range,
                "location": persona.estimated_location,
                "language": persona.estimated_language,
                "verified_ratio": float(persona.verified_ratio),
            },
            "engagement": {
                "avg_rate": float(persona.avg_engagement_rate),
                "potential_score": float(persona.engagement_potential_score),
                "growth_potential": float(persona.growth_potential_score),
                "retention_risk": float(persona.retention_risk_score),
            },
            "behavior": {
                "primary_interaction": persona.primary_interaction_type,
                "content_preferences": persona.content_type_preferences,
                "peak_hours": persona.peak_activity_hours,
                "peak_days": persona.peak_activity_days,
                "session_length_minutes": float(persona.avg_session_length_minutes),
                "posting_frequency": persona.posting_frequency_preference,
                "response_time_hours": float(persona.response_time_preference_hours),
            },
            "interests": {
                "primary": persona.primary_topics,
                "secondary": persona.secondary_topics,
                "avoided": persona.avoided_topics,
            },
            "recommendations": {
                "content_types": persona.recommended_content_types,
                "tone": persona.recommended_tone,
                "length": persona.recommended_length,
                "posting_times": persona.optimal_posting_times,
            },
        }

    def render_comparison(
        self, personas: List[PersonaProfile]
    ) -> Dict[str, Any]:
        """
        Render comparison of multiple personas.

        Args:
            personas: List of personas to compare

        Returns:
            Comparison data
        """
        if not personas:
            return {}

        return {
            "personas": [self.render_profile(p) for p in personas],
            "comparison": {
                "avg_engagement": sum(p.avg_engagement_rate for p in personas)
                / len(personas),
                "avg_size": sum(p.size for p in personas) / len(personas),
                "platforms": list(set(p.platform for p in personas)),
                "total_audience": sum(p.size for p in personas),
            },
        }