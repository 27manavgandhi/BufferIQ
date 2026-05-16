"""
Voice API models.

Pydantic models for voice analysis API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class VoiceExtractionRequest(BaseModel):
    """Request model for voice extraction."""
    
    brand_id: str = Field(..., description="Brand identifier")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")
    lookback_days: int = Field(90, ge=30, le=365, description="Days of history to analyze")
    min_posts: int = Field(20, ge=10, le=100, description="Minimum posts required")


class VoiceAnalysisRequest(BaseModel):
    """Request model for content voice analysis."""
    
    text: str = Field(..., min_length=10, description="Content to analyze")
    brand_id: str = Field(..., description="Brand identifier")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")
    return_recommendations: bool = Field(True, description="Include recommendations")
    return_validation: bool = Field(False, description="Include validation")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch content analysis."""
    
    contents: List[str] = Field(..., min_items=1, max_items=50, description="Content list")
    brand_id: str = Field(..., description="Brand identifier")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")


class VoiceValidationRequest(BaseModel):
    """Request model for voice validation."""
    
    text: str = Field(..., min_length=10, description="Content to validate")
    brand_id: str = Field(..., description="Brand identifier")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")


class DriftDetectionRequest(BaseModel):
    """Request model for drift detection."""
    
    brand_id: str = Field(..., description="Brand identifier")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")
    window_days: int = Field(30, ge=7, le=90, description="Recent window size in days")


class ConsistencyScoreResponse(BaseModel):
    """Consistency score details."""
    
    overall: float
    lexical: float
    syntactic: float
    stylistic: float
    is_consistent: bool
    severity: str


class RecommendationResponse(BaseModel):
    """Voice recommendation."""
    
    type: str
    priority: str
    current_value: str
    suggested_value: str
    reason: str
    impact_score: float
    examples: List[str]


class ValidationResponse(BaseModel):
    """Validation result."""
    
    passed: bool
    score: float
    threshold: float
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]


class VoiceAnalysisResponse(BaseModel):
    """Response model for voice analysis."""
    
    text: str
    brand_id: str
    platform: str
    profile_id: str
    consistency_score: ConsistencyScoreResponse
    metrics: Dict[str, float]
    recommendations: Optional[List[RecommendationResponse]] = None
    validation: Optional[ValidationResponse] = None
    analyzed_at: str


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    
    results: List[VoiceAnalysisResponse]
    total_analyzed: int
    successful: int
    failed: int


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile."""
    
    profile_id: str
    brand_id: str
    version: int
    created_at: str
    confidence: float
    sample_size: int
    signature: str
    platform_profiles: Dict[str, Dict]


class DriftAlertResponse(BaseModel):
    """Response model for drift detection."""
    
    brand_id: str
    platform: str
    drift_detected: bool
    drift_score: float
    drift_type: str
    affected_dimensions: List[str]
    severity: str
    statistical_tests: Dict[str, float]
    likely_causes: List[str]
    example_deviations: List[Dict]
    checked_at: str


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: Optional[str] = None
    timestamp: str