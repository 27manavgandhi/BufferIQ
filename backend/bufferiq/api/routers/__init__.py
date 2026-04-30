"""API routers."""

from bufferiq.api.routers import batch, health, metrics, models, prediction

__all__ = ["prediction", "batch", "models", "health", "metrics"]