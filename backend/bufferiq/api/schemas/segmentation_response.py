"""Response schemas for segmentation API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AnalysisMetadata(BaseModel):
    """Analysis metadata."""

    platform: str
    total_audience_size: int
    n_segments: int
    clustering_algorithm: str
    processing_time_ms: float


class SegmentMetrics(BaseModel):
    """Segment metrics."""

    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    stability_score: float


class PersonaData(BaseModel):
    """Persona response data."""

    segment_id: str
    persona_name: str
    persona_description: str
    size: int
    size_percentage: float


class RecommendationData(BaseModel):
    """Recommendation response data."""

    segment_id: str
    persona_name: str
    recommended_topics: List[str]
    recommended_formats: List[str]
    predicted_engagement_lift: float


class SegmentationAnalysisResponse(BaseModel):
    """Complete segmentation analysis response."""

    metadata: AnalysisMetadata
    metrics: SegmentMetrics
    personas: List[PersonaData]
    recommendations: List[RecommendationData]