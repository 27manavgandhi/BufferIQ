"""Health check endpoints."""

import time
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from bufferiq.api.dependencies import get_cache_service, get_model_loader
from bufferiq.api.models.health import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    ServiceHealth,
)
from bufferiq.api.services.cache_service import CacheService
from bufferiq.api.services.model_loader import ModelLoader

router = APIRouter()


async def check_cache_health(
    cache_service: CacheService,
) -> ServiceHealth:
    """Check cache service health."""
    try:
        start_time = time.time()
        # Simple ping test
        await cache_service.redis.ping()
        response_time_ms = (time.time() - start_time) * 1000

        return ServiceHealth(
            status="healthy",
            message="Cache is responding",
            response_time_ms=response_time_ms,
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"Cache error: {str(e)}",
        )


def check_models_health(model_loader: ModelLoader) -> ServiceHealth:
    """Check if models are loaded."""
    loaded_count = len(model_loader.models)
    total_count = len(model_loader.model_paths)

    if loaded_count == 0:
        return ServiceHealth(
            status="unhealthy",
            message="No models loaded",
        )

    return ServiceHealth(
        status="healthy",
        message=f"{loaded_count}/{total_count} models loaded",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    cache_service: CacheService = Depends(get_cache_service),
) -> HealthResponse:
    """
    Overall health check.

    Checks:
    - Cache connectivity
    - Models loaded

    Returns:
        Health status for all services
    """
    services: Dict[str, ServiceHealth] = {}

    # Check cache
    services["cache"] = await check_cache_health(cache_service)

    # Check models
    services["models"] = check_models_health(model_loader)

    # Overall status
    overall_status = (
        "healthy"
        if all(s.status == "healthy" for s in services.values())
        else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.utcnow(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    model_loader: ModelLoader = Depends(get_model_loader),
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Checks if the service is ready to accept requests.
    """
    # Check if at least one model is loaded
    if len(model_loader.models) == 0:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - no models loaded",
        )

    return ReadinessResponse(
        status="ready",
        message=f"{len(model_loader.models)} models loaded",
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe for Kubernetes.

    Always returns alive if the service is running.
    """
    return LivenessResponse(status="alive")