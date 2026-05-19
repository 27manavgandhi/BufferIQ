"""
API models for hashtag endpoints.

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HashtagAnalyzeRequest(BaseModel):
    """Request model for hashtag analysis."""

    hashtag: str = Field(..., description="Hashtag to analyze (without #)")
    platform: str = Field(..., description="Platform name (linkedin/twitter/bluesky)")
    user_id: Optional[str] = Field(None, description="Optional user context")


class HashtagPerformanceResponse(BaseModel):
    """Performance metrics response."""

    total_uses: int
    avg_engagement: float
    engagement_lift: float
    trend_direction: str
    roi: float


class HashtagRiskResponse(BaseModel):
    """Risk assessment response."""

    risk_level: str
    is_safe: bool
    reasons: List[str]
    recommendation: str


class RelatedHashtagResponse(BaseModel):
    """Related hashtag response."""

    hashtag: str
    score: float


class HashtagAnalyzeResponse(BaseModel):
    """Response model for hashtag analysis."""

    hashtag: str
    platform: str
    performance: HashtagPerformanceResponse
    risk: HashtagRiskResponse
    related: Dict[str, List[RelatedHashtagResponse]]


class HashtagRecommendRequest(BaseModel):
    """Request model for hashtag recommendations."""

    content: str = Field(..., description="Content text")
    platform: str = Field(..., description="Platform name")
    user_id: Optional[str] = Field(None, description="Optional user context")
    count: int = Field(5, ge=1, le=10, description="Number of recommendations")


class HashtagRecommendResponse(BaseModel):
    """Response model for recommendations."""

    hashtags: List[str]
    platform: str
    count: int


class TrendingHashtagRequest(BaseModel):
    """Request model for trending hashtags."""

    platform: str = Field(..., description="Platform name")
    category: Optional[str] = Field(None, description="Optional category filter")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")


class TrendingHashtagItem(BaseModel):
    """Trending hashtag item."""

    hashtag: str
    stage: str
    momentum_score: float
    current_volume: int
    volume_change: float
    recommendation: str


class TrendingHashtagResponse(BaseModel):
    """Response model for trending hashtags."""

    platform: str
    trending: List[TrendingHashtagItem]
    count: int


class HashtagDiscoverRequest(BaseModel):
    """Request model for hashtag discovery."""

    seed_hashtag: str = Field(..., description="Seed hashtag")
    platform: str = Field(..., description="Platform name")
    max_results: int = Field(20, ge=1, le=50, description="Max results per category")


class DiscoveredHashtagItem(BaseModel):
    """Discovered hashtag item."""

    hashtag: str
    similarity_score: float
    relationship_type: str
    effectiveness_score: float


class HashtagDiscoverResponse(BaseModel):
    """Response model for discovery."""

    seed_hashtag: str
    platform: str
    synonyms: List[DiscoveredHashtagItem]
    related: List[DiscoveredHashtagItem]
    complementary: List[DiscoveredHashtagItem]
    niche: List[DiscoveredHashtagItem]


class HashtagValidateRequest(BaseModel):
    """Request model for validation."""

    hashtags: List[str] = Field(..., description="Hashtags to validate")
    platform: str = Field(..., description="Platform name")


class HashtagValidationItem(BaseModel):
    """Validation result for single hashtag."""

    hashtag: str
    risk_level: str
    is_safe: bool
    recommendation: str
    reasons: List[str]


class HashtagValidateResponse(BaseModel):
    """Response model for validation."""

    platform: str
    results: Dict[str, HashtagValidationItem]


class HashtagInsightsResponse(BaseModel):
    """Detailed insights for a hashtag."""

    hashtag: str
    platform: str
    performance: HashtagPerformanceResponse
    risk: HashtagRiskResponse
    lifecycle_stage: str
    age_days: int
    related_hashtags: List[str]
    opportunities: List[str]