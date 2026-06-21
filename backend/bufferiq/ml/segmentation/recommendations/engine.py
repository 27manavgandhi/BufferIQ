"""Main recommendation engine."""

from typing import Any, Dict

from bufferiq.ml.segmentation.types import (
    PersonaProfile,
    SegmentRecommendation,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.segmentation.exceptions import UnsupportedPlatformError
from bufferiq.ml.segmentation.recommendations.content_recommender import (
    ContentRecommender,
)
from bufferiq.ml.segmentation.recommendations.timing_recommender import (
    TimingRecommender,
)
from bufferiq.ml.segmentation.recommendations.style_recommender import StyleRecommender
from bufferiq.ml.segmentation.recommendations.hashtag_recommender import (
    HashtagRecommender,
)


class RecommendationEngine:
    """
    Generate targeted recommendations for audience segments.

    Integrates:
    - Content recommendations
    - Timing recommendations
    - Style recommendations
    - Hashtag recommendations
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize recommendation engine."""
        self.config = config or {}
        self.content_recommender = ContentRecommender(
            self.config.get("content", {})
        )
        self.timing_recommender = TimingRecommender(self.config.get("timing", {}))
        self.style_recommender = StyleRecommender(self.config.get("style", {}))
        self.hashtag_recommender = HashtagRecommender(
            self.config.get("hashtag", {})
        )

    def generate(
        self,
        persona: PersonaProfile,
        platform: str,
    ) -> SegmentRecommendation:
        """
        Generate targeted recommendations.

        Args:
            persona: Persona profile
            platform: Platform type

        Returns:
            Targeted recommendation set

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        # Generate recommendations
        content_recs = self.content_recommender.recommend(persona)
        timing_recs = self.timing_recommender.recommend(persona)
        style_recs = self.style_recommender.recommend(persona)
        hashtag_recs = self.hashtag_recommender.recommend(persona)

        # Predict engagement lift
        lift, confidence = self._predict_lift(persona, content_recs)

        return SegmentRecommendation(
            segment_id=persona.segment_id,
            platform=platform,
            persona_name=persona.persona_name,
            recommended_topics=content_recs["topics"],
            recommended_formats=content_recs["formats"],
            recommended_tone=style_recs["tone"],
            recommended_length=style_recs["content_length"],
            sample_hooks=content_recs["sample_hooks"],
            optimal_posting_times=timing_recs["optimal_posting_times"],
            optimal_days=timing_recs["optimal_days"],
            posting_frequency=timing_recs["posting_frequency"],
            vocabulary_level=style_recs["vocabulary_level"],
            emoji_usage=style_recs["emoji_usage"],
            hashtag_count=hashtag_recs["hashtag_count"],
            recommended_hashtags=hashtag_recs["recommended_hashtags"],
            predicted_engagement_lift=lift,
            confidence_score=confidence,
        )

    def _predict_lift(
        self, persona: PersonaProfile, content_recs: Dict[str, Any]
    ) -> tuple[float, float]:
        """Predict engagement lift from recommendations."""
        # Base lift from engagement potential
        base_lift = persona.engagement_potential_score / 100.0 * 0.3

        # Format relevance
        format_lift = len(content_recs["formats"]) / 3.0 * 0.2

        # Topic relevance
        topic_lift = min(len(content_recs["topics"]) / 5.0 * 0.3, 0.3)

        total_lift = base_lift + format_lift + topic_lift

        # Confidence based on persona size
        confidence = min(persona.size / 100.0, 1.0)

        return (total_lift, confidence)