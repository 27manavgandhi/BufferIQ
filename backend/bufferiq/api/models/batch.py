"""Pydantic models for batch predictions."""

from typing import List

from pydantic import BaseModel, Field, validator

from bufferiq.api.models.prediction import (
    PredictionRequest,
    PredictionResponse,
)


class BatchItem(BaseModel):
    """Single item in batch request."""

    id: str = Field(..., description="Unique identifier for this item")
    request: PredictionRequest


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    items: List[BatchItem] = Field(..., min_items=1, max_items=100)

    @validator("items")
    def validate_batch_size(cls, v: List[BatchItem]) -> List[BatchItem]:
        """Validate batch size."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 items")
        return v

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "post_1",
                        "request": {
                            "content": "First post",
                            "platform": "linkedin",
                        },
                    },
                    {
                        "id": "post_2",
                        "request": {
                            "content": "Second post",
                            "platform": "twitter",
                        },
                    },
                ]
            }
        }


class BatchMetadata(BaseModel):
    """Metadata for batch predictions."""

    total_items: int
    processing_time_ms: float
    cache_hits: int = 0
    errors: int = 0


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""

    predictions: List[dict]  # List of {id, prediction or error}
    metadata: BatchMetadata

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "id": "post_1",
                        "prediction": {
                            "engagement_score": 7.5,
                            "confidence": 0.82,
                        },
                    },
                    {
                        "id": "post_2",
                        "prediction": {
                            "engagement_score": 6.2,
                            "confidence": 0.79,
                        },
                    },
                ],
                "metadata": {
                    "total_items": 2,
                    "processing_time_ms": 125.5,
                    "cache_hits": 0,
                    "errors": 0,
                },
            }
        }