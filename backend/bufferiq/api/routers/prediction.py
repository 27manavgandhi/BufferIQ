"""Prediction endpoints."""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from bufferiq.api.dependencies import (
    get_cache_service,
    get_monitoring_service,
    get_prediction_service,
)
from bufferiq.api.models.prediction import PredictionRequest, PredictionResponse
from bufferiq.api.services.cache_service import CacheService
from bufferiq.api.services.monitoring_service import MonitoringService
from bufferiq.api.services.prediction_service import PredictionService
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_engagement(
    request: PredictionRequest,
    model_name: Optional[str] = "ensemble",
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
    monitoring: MonitoringService = Depends(get_monitoring_service),
) -> PredictionResponse:
    """
    Predict engagement for a social media post.

    Args:
        request: Post content and metadata
        model_name: Model to use (default: ensemble)

    Returns:
        Prediction with confidence and breakdown

    Example: