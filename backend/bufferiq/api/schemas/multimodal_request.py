"""Request schemas for multi-modal API."""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class BaseAnalysisRequest(BaseModel):
    """Base request for analysis."""
    
    platform: str = Field(
        ...,
        description="Platform type (linkedin, twitter, or bluesky)",
        pattern="^(linkedin|twitter|bluesky)$"
    )


class ImageUploadRequest(BaseAnalysisRequest):
    """Request for image upload and analysis."""
    
    detect_objects: bool = Field(default=True)
    extract_text: bool = Field(default=True)
    analyze_faces: bool = Field(default=True)
    extract_colors: bool = Field(default=True)
    analyze_composition: bool = Field(default=True)
    score_aesthetics: bool = Field(default=True)
    detect_brand: bool = Field(default=True)


class BatchImageRequest(BaseAnalysisRequest):
    """Request for batch image analysis."""
    
    image_urls: List[HttpUrl] = Field(..., max_length=10)
    options: Optional[ImageUploadRequest] = None