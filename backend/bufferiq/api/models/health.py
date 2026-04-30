"""Pydantic models for health checks."""

from datetime import datetime
from typing import Dict, Literal

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    """Health status of individual service."""

    status: Literal["healthy", "unhealthy"]
    message: Optional[str] = None
    response_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Overall health check response."""

    status: Literal["healthy", "unhealthy"]
    services: Dict[str, ServiceHealth]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: Literal["ready", "not_ready"]
    message: Optional[str] = None


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: Literal["alive"]