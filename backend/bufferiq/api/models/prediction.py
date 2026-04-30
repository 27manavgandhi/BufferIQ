"""Pydantic models for predictions."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator

# Supported platforms
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class PredictionRequest(BaseModel):
    """Request for engagement prediction."""

    content: str = Field(..., min_length=1, max_length=10000)
    platform: Literal["linkedin", "twitter", "bluesky"]
    scheduled_time: Optional[datetime] = None
    post_type: Optional[str] = "text"
    has_media: bool = False
    has_link: bool = False

    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform is supported."""
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )
        return v

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "content": "Just shipped a new feature! 🚀",
                "platform": "linkedin",
                "scheduled_time": "2026-04-30T14:00:00Z",
                "post_type": "text",
                "has_media": False,
                "has_link": True,
            }
        }


class EngagementScores(BaseModel):
    """Breakdown of engagement metrics."""

    likes: float = Field(..., ge=0)
    comments: float = Field(..., ge=0)
    shares: float = Field(..., ge=0)


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_version: str
    inference_time_ms: float
    features_used: int
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""

    engagement_score: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    breakdown: EngagementScores
    metadata: PredictionMetadata

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "engagement_score": 7.8,
                "confidence": 0.85,
                "breakdown": {"likes": 45, "comments": 8, "shares": 3},
                "metadata": {
                    "model_version": "xgboost_v1.2.0",
                    "inference_time_ms": 45.2,
                    "features_used": 92,
                    "cached": False,
                    "timestamp": "2026-04-27T10:30:00Z",
                },
            }
        }