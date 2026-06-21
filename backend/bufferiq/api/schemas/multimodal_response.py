"""Response schemas for multi-modal API."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnalysisMetadata(BaseModel):
    """Metadata for analysis response."""
    
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    analyzer_version: str = Field(default="1.0.0", description="Analyzer version")


class ImageAnalysisResponse(BaseModel):
    """Response for image analysis."""
    
    analysis_id: str = Field(..., description="Analysis ID")
    platform: str = Field(..., description="Platform")
    objects_detected: int = Field(..., description="Number of objects detected")
    faces_detected: int = Field(..., description="Number of faces detected")
    text_elements: int = Field(..., description="Number of text elements")
    aesthetic_score: float = Field(..., description="Aesthetic quality score (0-100)")
    composition_scores: Dict[str, float] = Field(..., description="Composition scores")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")


class VideoAnalysisResponse(BaseModel):
    """Response for video analysis."""
    
    analysis_id: str = Field(..., description="Analysis ID")
    platform: str = Field(..., description="Platform")
    duration_seconds: float = Field(..., description="Video duration")
    resolution: List[int] = Field(..., description="Video resolution [width, height]")
    keyframe_count: int = Field(..., description="Number of keyframes extracted")
    scene_count: int = Field(..., description="Number of scenes detected")
    has_audio: bool = Field(..., description="Whether video has audio")
    engagement_prediction: float = Field(..., description="Predicted engagement (0-1)")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")


class LinkPreviewResponse(BaseModel):
    """Response for link preview analysis."""
    
    analysis_id: str = Field(..., description="Analysis ID")
    platform: str = Field(..., description="Platform")
    url: str = Field(..., description="Analyzed URL")
    title: Optional[str] = Field(None, description="Page title")
    description: Optional[str] = Field(None, description="Page description")
    image_url: Optional[str] = Field(None, description="Preview image URL")
    quality_scores: Dict[str, float] = Field(..., description="Quality scores")
    ctr_prediction: float = Field(..., description="Predicted CTR")
    optimization_suggestions: List[str] = Field(..., description="Optimization suggestions")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")