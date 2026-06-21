"""Pydantic models for multi-modal API."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl


class ImageAnalysisRequest(BaseModel):
    """Request for image analysis."""
    
    platform: Literal["linkedin", "twitter", "bluesky"] = Field(
        ...,
        description="Platform type (linkedin, twitter, or bluesky)"
    )
    detect_objects: bool = Field(default=True, description="Detect objects in image")
    extract_text: bool = Field(default=True, description="Extract text from image")
    analyze_faces: bool = Field(default=True, description="Analyze faces in image")
    detect_brand: bool = Field(default=True, description="Detect brand elements")


class VideoAnalysisRequest(BaseModel):
    """Request for video analysis."""
    
    video_url: HttpUrl = Field(..., description="URL to video file")
    platform: Literal["linkedin", "twitter", "bluesky"] = Field(
        ...,
        description="Platform type (linkedin, twitter, or bluesky)"
    )
    extract_keyframes: bool = Field(default=True, description="Extract key frames")
    detect_scenes: bool = Field(default=True, description="Detect scenes")
    analyze_audio: bool = Field(default=True, description="Analyze audio")


class LinkAnalysisRequest(BaseModel):
    """Request for link preview analysis."""
    
    url: HttpUrl = Field(..., description="URL to analyze")
    platform: Literal["linkedin", "twitter", "bluesky"] = Field(
        ...,
        description="Platform type (linkedin, twitter, or bluesky)"
    )


class MultiModalAnalysisRequest(BaseModel):
    """Request for complete multi-modal analysis."""
    
    post_id: str = Field(..., description="Post identifier")
    text: str = Field(..., description="Post text content")
    image_urls: Optional[List[HttpUrl]] = Field(default=None, description="Image URLs")
    video_urls: Optional[List[HttpUrl]] = Field(default=None, description="Video URLs")
    link_urls: Optional[List[HttpUrl]] = Field(default=None, description="Link URLs")
    platform: Literal["linkedin", "twitter", "bluesky"] = Field(
        ...,
        description="Platform type (linkedin, twitter, or bluesky)"
    )


class MultiModalAnalysisResponse(BaseModel):
    """Response for multi-modal analysis."""
    
    analysis_type: str = Field(..., description="Type of analysis performed")
    platform: str = Field(..., description="Platform analyzed for")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "analysis_type": "multi_modal",
                "platform": "linkedin",
                "results": {
                    "engagement_prediction": {
                        "predicted_engagement_rate": 0.087,
                        "improvement_potential": 23.5
                    }
                },
                "processing_time_ms": 1234.56
            }
        }