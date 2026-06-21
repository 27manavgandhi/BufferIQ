"""Request schemas for segmentation API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseSegmentationRequest(BaseModel):
    """Base segmentation request."""

    platform: str

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform."""
        if v not in ["linkedin", "twitter", "bluesky"]:
            raise ValueError("Platform not supported")
        return v


class AudienceDataRequest(BaseSegmentationRequest):
    """Request with audience data."""

    audience_data: List[Dict[str, Any]]

    @field_validator("audience_data")
    @classmethod
    def validate_data(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate audience data."""
        if len(v) < 10:
            raise ValueError("Need at least 10 audience members")
        return v


class BatchSegmentationRequest(BaseModel):
    """Batch segmentation request."""

    requests: List[AudienceDataRequest]

    @field_validator("requests")
    @classmethod
    def validate_requests(cls, v: List[AudienceDataRequest]) -> List[AudienceDataRequest]:
        """Validate requests."""
        if not v:
            raise ValueError("requests cannot be empty")
        return v