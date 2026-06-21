"""Main persona builder orchestrating all components."""

from typing import Any, Dict, List

import numpy as np
from datetime import datetime

from bufferiq.ml.segmentation.types import PersonaProfile, AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    PersonaGenerationError,
    UnsupportedPlatformError,
)
from bufferiq.ml.segmentation.personas.demographic_inferrer import DemographicInferrer
from bufferiq.ml.segmentation.personas.behavioral_profiler import BehavioralProfiler
from bufferiq.ml.segmentation.personas.interest_mapper import InterestMapper
from bufferiq.ml.segmentation.personas.content_preference_modeler import (
    ContentPreferenceModeler,
)
from bufferiq.ml.segmentation.personas.namer import PersonaNamer


class PersonaBuilder:
    """
    Build detailed persona profiles from cluster data.

    Orchestrates:
    - Demographic inference
    - Behavioral profiling
    - Interest mapping
    - Content preference modeling
    - Persona naming
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize persona builder."""
        self.config = config or {}
        self.demographic_inferrer = DemographicInferrer(
            self.config.get("demographic", {})
        )
        self.behavioral_profiler = BehavioralProfiler(self.config.get("behavioral", {}))
        self.interest_mapper = InterestMapper(self.config.get("interest", {}))
        self.content_preference_modeler = ContentPreferenceModeler(
            self.config.get("content_preference", {})
        )
        self.namer = PersonaNamer(self.config.get("namer", {}))

    def build(
        self,
        cluster_id: int,
        cluster_members: List[AudienceDataPoint],
        feature_matrix: np.ndarray,
        platform: str,
    ) -> PersonaProfile:
        """
        Build complete persona profile for a cluster.

        Args:
            cluster_id: Cluster identifier
            cluster_members: List of audience members in cluster
            feature_matrix: Feature vectors for cluster members
            platform: Platform type

        Returns:
            Complete persona profile

        Raises:
            UnsupportedPlatformError: If platform not supported
            PersonaGenerationError: If generation fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if not cluster_members:
            raise PersonaGenerationError("cluster_members cannot be empty")

        try:
            segment_id = f"segment_{cluster_id}_{platform}"
            total_size = len(cluster_members)

            # Infer demographics
            demographics = self.demographic_inferrer.infer(cluster_members, platform)

            # Profile behaviors
            behaviors = self.behavioral_profiler.profile(
                cluster_members, feature_matrix, platform
            )

            # Map interests
            interests = self.interest_mapper.map(cluster_members, platform)

            # Model content preferences
            content_prefs = self.content_preference_modeler.model(cluster_members)

            # Compute engagement metrics
            engagement_metrics = self._compute_engagement_metrics(
                cluster_members, feature_matrix
            )

            # Generate persona name and description
            persona_name = self.namer.generate_name(
                demographics, behaviors, interests, platform
            )
            persona_description = self.namer.generate_description(
                demographics, behaviors, interests, platform
            )

            # Build recommendations
            content_recs = self._build_content_recommendations(
                behaviors, interests, platform
            )

            return PersonaProfile(
                segment_id=segment_id,
                platform=platform,
                persona_name=persona_name,
                persona_description=persona_description,
                size=total_size,
                size_percentage=0.0,  # Set externally
                estimated_age_range=demographics["age_range"],
                estimated_location=demographics.get("location"),
                estimated_language=demographics.get("language", "en"),
                verified_ratio=demographics["verified_ratio"],
                avg_engagement_rate=engagement_metrics["avg_engagement_rate"],
                primary_interaction_type=behaviors["primary_interaction"],
                content_type_preferences=content_prefs["formats"],
                peak_activity_hours=behaviors["peak_hours"],
                peak_activity_days=behaviors["peak_days"],
                primary_topics=interests["primary"],
                secondary_topics=interests["secondary"],
                avoided_topics=interests["avoided"],
                avg_session_length_minutes=behaviors["avg_session_length"],
                posting_frequency_preference=behaviors["posting_frequency"],
                response_time_preference_hours=behaviors["response_time"],
                recommended_content_types=content_recs["content_types"],
                recommended_tone=content_recs["tone"],
                recommended_length=content_recs["length"],
                optimal_posting_times=content_recs["posting_times"],
                engagement_potential_score=engagement_metrics["potential_score"],
                growth_potential_score=engagement_metrics["growth_score"],
                retention_risk_score=engagement_metrics["retention_risk"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        except (UnsupportedPlatformError, ValueError) as e:
            raise e
        except Exception as e:
            raise PersonaGenerationError(f"Persona generation failed: {str(e)}")

    def _compute_engagement_metrics(
        self,
        cluster_members: List[AudienceDataPoint],
        feature_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Compute engagement-related metrics."""
        avg_engagement = np.mean([m.avg_engagement_rate for m in cluster_members])

        # Potential score based on verified ratio and follower count
        verified_ratio = sum(1 for m in cluster_members if m.verified) / len(
            cluster_members
        )
        avg_followers = np.mean([m.follower_count for m in cluster_members])

        potential_score = min(
            (verified_ratio * 0.5 + (avg_followers / 10000.0) * 0.5) * 100, 100.0
        )

        # Growth potential based on following ratio
        avg_following_ratio = np.mean(
            [m.follower_count / max(m.following_count, 1) for m in cluster_members]
        )
        growth_score = min((1.0 - avg_following_ratio) * 100, 100.0)

        # Retention risk (inverse of engagement)
        retention_risk = (1.0 - avg_engagement) * 100

        return {
            "avg_engagement_rate": float(avg_engagement),
            "potential_score": float(potential_score),
            "growth_score": float(growth_score),
            "retention_risk": float(retention_risk),
        }

    def _build_content_recommendations(
        self, behaviors: Dict[str, Any], interests: Dict[str, List[str]], platform: str
    ) -> Dict[str, Any]:
        """Build content recommendations for the persona."""
        return {
            "content_types": list(
                sorted(
                    behaviors["content_preferences"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
            ) if behaviors["content_preferences"] else ["text"],
            "tone": "professional" if platform == "linkedin" else "casual",
            "length": behaviors.get("length_preference", "medium"),
            "posting_times": [
                f"{h:02d}:00" for h in behaviors.get("peak_hours", [9, 12, 18])
            ],
        }