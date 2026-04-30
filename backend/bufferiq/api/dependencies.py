"""Dependency injection for FastAPI."""

from typing import AsyncGenerator, Generator

from fastapi import Depends

from bufferiq.api.services.cache_service import CacheService
from bufferiq.api.services.feature_service import FeatureService
from bufferiq.api.services.model_loader import ModelLoader
from bufferiq.api.services.monitoring_service import MonitoringService
from bufferiq.api.services.prediction_service import PredictionService
from bufferiq.core.config import get_settings

settings = get_settings()


def get_model_loader() -> ModelLoader:
    """Get model loader singleton."""
    return ModelLoader()


async def get_cache_service() -> AsyncGenerator[CacheService, None]:
    """Get cache service."""
    service = CacheService(
        redis_url=settings.redis_url, ttl=settings.cache_ttl
    )
    try:
        yield service
    finally:
        await service.close()


def get_feature_service() -> FeatureService:
    """Get feature service."""
    return FeatureService()


def get_monitoring_service() -> MonitoringService:
    """Get monitoring service."""
    return MonitoringService()


def get_prediction_service(
    model_loader: ModelLoader = Depends(get_model_loader),
    feature_service: FeatureService = Depends(get_feature_service),
) -> PredictionService:
    """Get prediction service."""
    return PredictionService(
        model_loader=model_loader, feature_service=feature_service
    )