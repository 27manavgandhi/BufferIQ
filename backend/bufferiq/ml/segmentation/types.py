"""Data types for audience segmentation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class AudienceDataPoint:
    """Single audience member data point."""

    user_id: str
    platform: str
    follower_count: int
    following_count: int
    post_count: int
    avg_engagement_rate: float
    engagement_history: List[Dict[str, Any]] = field(default_factory=list)
    interaction_types: Dict[str, int] = field(default_factory=dict)
    active_hours: List[int] = field(default_factory=list)
    active_days: List[int] = field(default_factory=list)
    topics_engaged: List[str] = field(default_factory=list)
    content_types_engaged: List[str] = field(default_factory=list)
    account_age_days: int = 0
    verified: bool = False
    bio_keywords: List[str] = field(default_factory=list)
    location: Optional[str] = None
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate platform."""
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{self.platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "platform": self.platform,
            "follower_count": self.follower_count,
            "following_count": self.following_count,
            "post_count": self.post_count,
            "avg_engagement_rate": self.avg_engagement_rate,
            "verified": self.verified,
            "account_age_days": self.account_age_days,
        }


@dataclass
class ProcessedAudienceFeatures:
    """Processed feature vector for a single audience member."""

    user_id: str
    platform: str
    feature_vector: np.ndarray
    feature_names: List[str]
    raw_features: Dict[str, float]
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "platform": self.platform,
            "feature_vector_shape": self.feature_vector.shape,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "raw_features_keys": list(self.raw_features.keys()),
            "processing_timestamp": self.processing_timestamp.isoformat(),
        }


@dataclass
class ClusteringResult:
    """Result of clustering operation."""

    algorithm: str
    n_clusters: int
    labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    convergence_iterations: Optional[int]
    noise_ratio: float
    stability_score: float
    platform: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "silhouette_score": float(self.silhouette_score),
            "calinski_harabasz_score": float(self.calinski_harabasz_score),
            "davies_bouldin_score": float(self.davies_bouldin_score),
            "inertia": float(self.inertia) if self.inertia is not None else None,
            "convergence_iterations": self.convergence_iterations,
            "noise_ratio": float(self.noise_ratio),
            "stability_score": float(self.stability_score),
            "platform": self.platform,
        }


@dataclass
class OptimalClusterConfig:
    """Optimal clustering configuration."""

    algorithm: str
    n_clusters: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "parameters": self.parameters,
            "validation_scores": {
                k: float(v) for k, v in self.validation_scores.items()
            },
            "confidence": float(self.confidence),
        }


@dataclass
class PersonaProfile:
    """Complete persona profile for an audience segment."""

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
    content_type_preferences: Dict[str, float] = field(default_factory=dict)
    peak_activity_hours: List[int] = field(default_factory=list)
    peak_activity_days: List[str] = field(default_factory=list)
    primary_topics: List[str] = field(default_factory=list)
    secondary_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    avg_session_length_minutes: float = 0.0
    posting_frequency_preference: str = "weekly"
    response_time_preference_hours: float = 24.0
    recommended_content_types: List[str] = field(default_factory=list)
    recommended_tone: str = "professional"
    recommended_length: str = "medium"
    optimal_posting_times: List[str] = field(default_factory=list)
    engagement_potential_score: float = 0.0
    growth_potential_score: float = 0.0
    retention_risk_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "platform": self.platform,
            "persona_name": self.persona_name,
            "persona_description": self.persona_description,
            "size": self.size,
            "size_percentage": float(self.size_percentage),
            "estimated_age_range": self.estimated_age_range,
            "estimated_location": self.estimated_location,
            "estimated_language": self.estimated_language,
            "verified_ratio": float(self.verified_ratio),
            "avg_engagement_rate": float(self.avg_engagement_rate),
            "primary_interaction_type": self.primary_interaction_type,
            "primary_topics": self.primary_topics,
            "secondary_topics": self.secondary_topics,
            "avoided_topics": self.avoided_topics,
            "engagement_potential_score": float(self.engagement_potential_score),
            "growth_potential_score": float(self.growth_potential_score),
            "retention_risk_score": float(self.retention_risk_score),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SegmentSnapshot:
    """Point-in-time snapshot of a segment."""

    segment_id: str
    platform: str
    timestamp: datetime
    size: int
    avg_engagement_rate: float
    member_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    health_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "platform": self.platform,
            "timestamp": self.timestamp.isoformat(),
            "size": self.size,
            "avg_engagement_rate": float(self.avg_engagement_rate),
            "health_score": float(self.health_score),
            "metrics_keys": list(self.metrics.keys()),
        }


@dataclass
class SegmentEvolution:
    """Tracks how a segment evolves over time."""

    segment_id: str
    platform: str
    snapshots: List[SegmentSnapshot] = field(default_factory=list)
    growth_rate: float = 0.0
    engagement_trend: str = "stable"
    stability_score: float = 0.0
    predicted_size_30d: int = 0
    predicted_engagement_30d: float = 0.0
    migration_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "platform": self.platform,
            "n_snapshots": len(self.snapshots),
            "growth_rate": float(self.growth_rate),
            "engagement_trend": self.engagement_trend,
            "stability_score": float(self.stability_score),
            "predicted_size_30d": self.predicted_size_30d,
            "predicted_engagement_30d": float(self.predicted_engagement_30d),
        }


@dataclass
class SegmentRecommendation:
    """Targeted recommendation for a specific segment."""

    segment_id: str
    platform: str
    persona_name: str
    recommended_topics: List[str] = field(default_factory=list)
    recommended_formats: List[str] = field(default_factory=list)
    recommended_tone: str = "professional"
    recommended_length: str = "medium"
    sample_hooks: List[str] = field(default_factory=list)
    optimal_posting_times: List[str] = field(default_factory=list)
    optimal_days: List[str] = field(default_factory=list)
    posting_frequency: str = "weekly"
    vocabulary_level: str = "moderate"
    emoji_usage: str = "minimal"
    hashtag_count: int = 5
    recommended_hashtags: List[str] = field(default_factory=list)
    predicted_engagement_lift: float = 0.0
    confidence_score: float = 0.0
    suggested_variants: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "platform": self.platform,
            "persona_name": self.persona_name,
            "recommended_topics": self.recommended_topics,
            "recommended_formats": self.recommended_formats,
            "recommended_tone": self.recommended_tone,
            "recommended_length": self.recommended_length,
            "vocabulary_level": self.vocabulary_level,
            "predicted_engagement_lift": float(self.predicted_engagement_lift),
            "confidence_score": float(self.confidence_score),
            "n_variants": len(self.suggested_variants),
        }