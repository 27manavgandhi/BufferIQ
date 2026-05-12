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

    Example:POST /api/v1/predict
    {
      "content": "Just shipped a new feature!",
      "platform": "linkedin",
      "scheduled_time": "2026-04-30T14:00:00Z"
    }
    """
    start_time = time.time()

    # Increment request counter
    monitoring.increment_request_count("predict", request.platform)

    try:
        # Check cache
        cache_key = cache_service.generate_key(request, model_name)
        cached_response = await cache_service.get(cache_key)

        if cached_response:
            monitoring.increment_cache_hits()
            cached_response.metadata.cached = True
            return cached_response

        # Make prediction
        response = await prediction_service.predict(request, model_name)

        # Cache response
        await cache_service.set(cache_key, response)

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        monitoring.record_latency("predict", duration_ms)
        response.metadata.inference_time_ms = duration_ms

        return response

    except ValueError as e:
        monitoring.increment_error_count("predict", "validation_error")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        monitoring.increment_error_count("predict", "internal_error")
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_with_ensemble(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
) -> PredictionResponse:
    """
    Predict engagement using ensemble model.

    This endpoint explicitly uses the ensemble model.
    """
    return await predict_engagement(
        request=request,
        model_name="ensemble",
        prediction_service=prediction_service,
        cache_service=cache_service,
        monitoring=get_monitoring_service(),
    )