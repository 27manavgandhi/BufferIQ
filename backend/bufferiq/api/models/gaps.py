"""API models for gap analysis."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class GapAnalysisRequest(BaseModel):
    """Request model for gap analysis."""

    user_id: str = Field(..., description="User identifier")
    platform: str = Field(..., description="Platform to analyze")
    competitor_ids: Optional[List[str]] = Field(None, description="Competitor IDs")
    industry: Optional[str] = Field(None, description="Industry category")
    lookback_days: int = Field(90, ge=7, le=365, description="Days of history")
    include_recommendations: bool = Field(True, description="Include recommendations")

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        allowed = ["linkedin", "twitter", "bluesky"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of: {allowed}")
        return v


class GapAnalysisResponse(BaseModel):
    """Response model for gap analysis."""

    user_id: str
    platform: str
    analysis_date: str
    lookback_days: int
    topics_found: int
    coverage_score: float
    total_gaps: int
    critical_gaps: List[Dict[str, Any]]
    important_gaps: List[Dict[str, Any]]
    recommendations_count: int
    competitive_position: str


class CalendarRequest(BaseModel):
    """Request model for calendar generation."""

    user_id: str = Field(..., description="User identifier")
    platform: str = Field(..., description="Target platform")
    weeks: int = Field(4, ge=1, le=12, description="Number of weeks")
    posts_per_week: int = Field(3, ge=1, le=7, description="Posts per week")
    start_date: Optional[datetime] = Field(None, description="Start date")

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        allowed = ["linkedin", "twitter", "bluesky"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of: {allowed}")
        return v


class CalendarResponse(BaseModel):
    """Response model for calendar."""

    start_date: str
    end_date: str
    total_pieces: int
    calendar_items: List[Dict[str, Any]]
    posting_frequency: float


class RecommendationsRequest(BaseModel):
    """Request model for content recommendations."""

    user_id: str = Field(..., description="User identifier")
    platform: str = Field(..., description="Platform")
    count: int = Field(10, ge=1, le=50, description="Number of recommendations")

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        allowed = ["linkedin", "twitter", "bluesky"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of: {allowed}")
        return v


class CompetitorAnalysisRequest(BaseModel):
    """Request model for competitor analysis."""

    user_id: str = Field(..., description="User identifier")
    competitor_ids: List[str] = Field(..., description="Competitor IDs")
    platform: str = Field(..., description="Platform")
    lookback_days: int = Field(90, ge=7, le=365, description="Days of history")

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        allowed = ["linkedin", "twitter", "bluesky"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of: {allowed}")
        return v

    @validator("competitor_ids")
    def validate_competitors(cls, v: List[str]) -> List[str]:
        """Validate competitor list."""
        if len(v) < 1:
            raise ValueError("At least 1 competitor required")
        if len(v) > 10:
            raise ValueError("Maximum 10 competitors allowed")
        return v


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""

    user_ids: List[str] = Field(..., description="List of user IDs")
    platform: str = Field(..., description="Platform")
    lookback_days: int = Field(90, description="Days of history")

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        allowed = ["linkedin", "twitter", "bluesky"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of: {allowed}")
        return v

    @validator("user_ids")
    def validate_user_ids(cls, v: List[str]) -> List[str]:
        """Validate user ID list."""
        if len(v) < 1:
            raise ValueError("At least 1 user ID required")
        if len(v) > 100:
            raise ValueError("Maximum 100 user IDs allowed")
        return v