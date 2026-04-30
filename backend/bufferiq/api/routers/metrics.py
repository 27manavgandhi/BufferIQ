"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Depends, Response

from bufferiq.api.dependencies import get_monitoring_service
from bufferiq.api.services.monitoring_service import MonitoringService

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics(
    monitoring: MonitoringService = Depends(get_monitoring_service),
) -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    metrics_data = generate_latest(monitoring.registry)

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST,
    )