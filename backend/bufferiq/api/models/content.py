"""
Content analysis API models.

Pydantic models for content analysis endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""

    text: str = Field(..., min_length=1, max_length=10000)
    platform: str = Field(..., description="Platform type")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    include_optimization: bool = Field(
        True, description="Include optimization suggestions"
    )

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform is supported."""
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. Supported: {SUPPORTED_PLATFORMS}"
            )
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class SentimentResponse(BaseModel):
    """Sentiment analysis response."""

    sentiment: str
    confidence: float
    polarity: float
    subjectivity: float
    scores: Dict[str, float]


class ReadabilityResponse(BaseModel):
    """Readability analysis response."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    average_grade_level: float
    reading_difficulty: str


class QualityResponse(BaseModel):
    """Quality check response."""

    score: float
    grammar_errors: int
    spelling_errors: int
    broken_links: int
    warnings: List[str]
    recommendations: List[str]


class OptimizationSuggestion(BaseModel):
    """Optimization suggestion."""

    type: str
    priority: str
    current_value: Any
    suggested_value: Any
    impact: str
    confidence: float


class OptimizationResponse(BaseModel):
    """Optimization response."""

    overall_score: float
    predicted_engagement_lift: float
    best_platform: str
    suggestions: List[OptimizationSuggestion]
    rewrite_examples: List[str]


class ContentAnalysisResponse(BaseModel):
    """Complete content analysis response."""

    text: str
    platform: str
    preprocessed: Dict[str, Any]
    features: Dict[str, Any]
    sentiment: SentimentResponse
    quality: QualityResponse
    readability: Optional[ReadabilityResponse] = None
    optimization: Optional[OptimizationResponse] = None
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""

    posts: List[Dict[str, str]] = Field(..., min_length=1, max_length=100)
    platform: str

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform is supported."""
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. Supported: {SUPPORTED_PLATFORMS}"
            )
        return v


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""

    results: List[ContentAnalysisResponse]
    total_analyzed: int
    total_errors: int