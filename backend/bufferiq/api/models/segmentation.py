"""Pydantic models for segmentation API."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class AudienceDataPointModel(BaseModel):
    """API model for audience data point."""

    user_id: str
    platform: str
    follower_count: int
    following_count: int
    post_count: int
    avg_engagement_rate: float
    engagement_history: List[Dict[str, Any]] = Field(default_factory=list)
    interaction_types: Dict[str, int] = Field(default_factory=dict)
    active_hours: List[int] = Field(default_factory=list)
    active_days: List[int] = Field(default_factory=list)
    topics_engaged: List[str] = Field(default_factory=list)
    content_types_engaged: List[str] = Field(default_factory=list)
    account_age_days: int = 0
    verified: bool = False
    bio_keywords: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    language: str = "en"

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        if v not in ["linkedin", "twitter", "bluesky"]:
            raise ValueError(
                f"Platform '{v}' not supported. "
                f"Supported: linkedin, twitter, bluesky"
            )
        return v

    @field_validator("avg_engagement_rate")
    @classmethod
    def validate_engagement(cls, v: float) -> float:
        """Validate engagement rate."""
        if not (0 <= v <= 1):
            raise ValueError("avg_engagement_rate must be between 0 and 1")
        return v


class SegmentAudienceRequest(BaseModel):
    """Request to segment audience."""

    audience_data: List[AudienceDataPointModel]
    platform: str
    historical_snapshots: Optional[Dict[str, List[Dict[str, Any]]]] = None

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        if v not in ["linkedin", "twitter", "bluesky"]:
            raise ValueError(
                f"Platform '{v}' not supported. "
                f"Supported: linkedin, twitter, bluesky"
            )
        return v

    @field_validator("audience_data")
    @classmethod
    def validate_audience_data(cls, v: List[AudienceDataPointModel]) -> List[AudienceDataPointModel]:
        """Validate audience data."""
        if len(v) < 10:
            raise ValueError("Need at least 10 audience members for segmentation")
        return v


class PersonaResponse(BaseModel):
    """Response with persona data."""

    segment_id: str
    platform: str
    persona_name: str
    persona_description: str
    size: int
    size_percentage: float
    estimated_age_range: Tuple[int, int]
    estimated_location: Optional[str]
    estimated_language: str
    verified_ratio: float
    avg_engagement_rate: float
    primary_interaction_type: str
    content_type_preferences: Dict[str, float]
    peak_activity_hours: List[int]
    peak_activity_days: List[str]
    primary_topics: List[str]
    secondary_topics: List[str]
    avoided_topics: List[str]
    engagement_potential_score: float
    growth_potential_score: float
    retention_risk_score: float


class RecommendationResponse(BaseModel):
    """Response with recommendations."""

    segment_id: str
    platform: str
    persona_name: str
    recommended_topics: List[str]
    recommended_formats: List[str]
    recommended_tone: str
    recommended_length: str
    sample_hooks: List[str]
    optimal_posting_times: List[str]
    optimal_days: List[str]
    posting_frequency: str
    vocabulary_level: str
    emoji_usage: str
    hashtag_count: int
    recommended_hashtags: List[str]
    predicted_engagement_lift: float
    confidence_score: float


class ClusteringQualityResponse(BaseModel):
    """Clustering quality metrics."""

    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    stability_score: float


class SegmentationResponse(BaseModel):
    """Complete segmentation response."""

    platform: str
    total_audience_size: int
    n_segments: int
    clustering_algorithm: str
    clustering_quality: ClusteringQualityResponse
    personas: List[PersonaResponse]
    recommendations: List[RecommendationResponse]
    processing_time_ms: float
    segmented_at: str