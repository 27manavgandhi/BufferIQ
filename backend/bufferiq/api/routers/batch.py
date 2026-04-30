"""Batch prediction endpoints."""

import asyncio
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from bufferiq.api.dependencies import get_cache_service, get_prediction_service
from bufferiq.api.models.batch import (
    BatchMetadata,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from bufferiq.api.services.cache_service import CacheService
from bufferiq.api.services.prediction_service import PredictionService
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/batch/predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    model_name: str = "ensemble",
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
) -> BatchPredictionResponse:
    """
    Predict engagement for multiple posts in batch.

    Processes up to 100 posts concurrently.

    Args:
        request: Batch of prediction requests
        model_name: Model to use

    Returns:
        Batch predictions with metadata
    """
    start_time = time.time()
    cache_hits = 0
    errors = 0
    predictions = []

    # Process batch concurrently
    async def process_item(item):
        nonlocal cache_hits, errors

        try:
            # Check cache
            cache_key = cache_service.generate_key(item.request, model_name)
            cached = await cache_service.get(cache_key)

            if cached:
                cache_hits += 1
                return {
                    "id": item.id,
                    "prediction": cached.dict(),
                }

            # Make prediction
            prediction = await prediction_service.predict(
                item.request, model_name
            )

            # Cache result
            await cache_service.set(cache_key, prediction)

            return {
                "id": item.id,
                "prediction": prediction.dict(),
            }

        except Exception as e:
            errors += 1
            logger.error(f"Batch item {item.id} failed: {e}")
            return {
                "id": item.id,
                "error": str(e),
            }

    # Process all items concurrently
    predictions = await asyncio.gather(
        *[process_item(item) for item in request.items]
    )

    # Calculate metadata
    processing_time_ms = (time.time() - start_time) * 1000
    metadata = BatchMetadata(
        total_items=len(request.items),
        processing_time_ms=processing_time_ms,
        cache_hits=cache_hits,
        errors=errors,
    )

    return BatchPredictionResponse(predictions=predictions, metadata=metadata)