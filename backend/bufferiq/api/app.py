"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bufferiq.api.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    TimingMiddleware,
)
from bufferiq.api.routers import batch, health, metrics, models, prediction
from bufferiq.api.services.model_loader import ModelLoader
from bufferiq.core.config import get_settings
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting BufferIQ API...")

    # Initialize model loader
    model_loader = ModelLoader()

    # Register models
    model_loader.register_model(
        "xgboost", settings.model_path / "xgboost_best.joblib"
    )
    model_loader.register_model(
        "lightgbm", settings.model_path / "lightgbm_best.joblib"
    )
    model_loader.register_model(
        "random_forest", settings.model_path / "random_forest_best.joblib"
    )
    model_loader.register_model(
        "ensemble", settings.model_path / "ensembles/production_ensemble.joblib"
    )

    # Warmup models
    if settings.warmup_models:
        model_loader.warmup()

    logger.info("BufferIQ API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down BufferIQ API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="BufferIQ Prediction API",
        description="ML-powered social media engagement prediction",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Include routers
    app.include_router(prediction.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(batch.router, prefix="/api/v1", tags=["Batch"])
    app.include_router(models.router, prefix="/api/v1", tags=["Models"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(metrics.router, tags=["Metrics"])

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        """Root endpoint with API information."""
        return {
            "name": "BufferIQ API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle all unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    return app


# Create app instance
app = create_app()