"""Service layer for segmentation API."""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from bufferiq.ml.segmentation.intelligence.service import SegmentationIntelligenceService
from bufferiq.ml.segmentation.types import AudienceDataPoint, SegmentSnapshot
from bufferiq.domain.repositories.segment_repository import (
    SegmentRepository,
    PersonaRepository,
    EvolutionRepository,
)


class SegmentationAPIService:
    """Service layer orchestrating segmentation."""

    def __init__(
        self,
        intelligence_service: SegmentationIntelligenceService,
        db: Session,
    ) -> None:
        """Initialize API service."""
        self.intelligence_service = intelligence_service
        self.db = db
        self.segment_repo = SegmentRepository(db)
        self.persona_repo = PersonaRepository(db)
        self.evolution_repo = EvolutionRepository(db)

    async def segment_and_persist(
        self,
        audience_data: List[AudienceDataPoint],
        platform: str,
    ) -> Dict[str, Any]:
        """
        Segment audience and persist results.

        Args:
            audience_data: List of audience data points
            platform: Platform type

        Returns:
            Segmentation result
        """
        # Run segmentation
        result = await self.intelligence_service.segment_audience(
            audience_data=audience_data,
            platform=platform,
        )

        # Persist segments
        for persona_dict in result.get("personas", []):
            segment_data = {
                "id": persona_dict["segment_id"],
                "platform": platform,
                "n_members": persona_dict["size"],
                "size_percentage": persona_dict["size_percentage"],
                "clustering_algorithm": result.get("clustering_algorithm", "kmeans"),
                "silhouette_score": result.get("clustering_quality", {}).get(
                    "silhouette_score"
                ),
                "stability_score": result.get("clustering_quality", {}).get(
                    "stability_score"
                ),
            }

            self.segment_repo.create_segment(segment_data)

            # Persist persona
            persona_data = {
                "id": f"persona_{persona_dict['segment_id']}",
                "segment_id": persona_dict["segment_id"],
                "platform": platform,
                "persona_name": persona_dict["persona_name"],
                "persona_description": persona_dict["persona_description"],
                "estimated_age_min": persona_dict["estimated_age_range"][0],
                "estimated_age_max": persona_dict["estimated_age_range"][1],
                "estimated_location": persona_dict.get("estimated_location"),
                "verified_ratio": persona_dict.get("verified_ratio"),
                "avg_engagement_rate": persona_dict.get("avg_engagement_rate"),
                "primary_interaction_type": persona_dict.get(
                    "primary_interaction_type"
                ),
                "content_preferences": persona_dict.get("content_type_preferences"),
                "peak_activity_hours": persona_dict.get("peak_activity_hours"),
                "peak_activity_days": persona_dict.get("peak_activity_days"),
                "primary_topics": persona_dict.get("primary_topics"),
                "secondary_topics": persona_dict.get("secondary_topics"),
                "avoided_topics": persona_dict.get("avoided_topics"),
                "engagement_potential_score": persona_dict.get(
                    "engagement_potential_score"
                ),
                "growth_potential_score": persona_dict.get("growth_potential_score"),
                "retention_risk_score": persona_dict.get("retention_risk_score"),
            }

            self.persona_repo.create_persona(persona_data)

        return result

    async def get_recommendations(
        self, segment_id: str, platform: str
    ) -> Dict[str, Any]:
        """Get recommendations for a segment."""
        persona = self.persona_repo.get_personas_by_segment(segment_id)

        if not persona:
            return {}

        recommendation = {
            "segment_id": segment_id,
            "platform": platform,
            "persona_name": persona[0].persona_name,
            "recommended_topics": persona[0].primary_topics or [],
            "recommended_formats": list(
                (persona[0].content_preferences or {}).keys()
            )[:3],
            "recommended_tone": "professional"
            if platform == "linkedin"
            else "casual",
            "recommended_length": "medium",
            "sample_hooks": [],
            "optimal_posting_times": [],
            "optimal_days": persona[0].peak_activity_days or [],
            "posting_frequency": "weekly",
            "vocabulary_level": "moderate",
            "emoji_usage": "minimal",
            "hashtag_count": 5,
            "recommended_hashtags": [],
            "predicted_engagement_lift": 0.15,
            "confidence_score": 0.85,
        }

        return recommendation