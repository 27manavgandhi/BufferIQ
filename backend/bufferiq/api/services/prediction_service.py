"""Prediction service orchestrating the ML pipeline."""

import numpy as np
from typing import Optional

from bufferiq.api.models.prediction import (
    EngagementScores,
    PredictionMetadata,
    PredictionRequest,
    PredictionResponse,
)
from bufferiq.api.services.feature_service import FeatureService
from bufferiq.api.services.model_loader import ModelLoader
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    Service for making predictions.

    Orchestrates the full prediction pipeline:
    1. Extract features from request
    2. Load appropriate model
    3. Make prediction
    4. Format response with metadata
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        feature_service: FeatureService,
    ) -> None:
        """
        Initialize prediction service.

        Args:
            model_loader: Model loader instance
            feature_service: Feature extraction service
        """
        self.model_loader = model_loader
        self.feature_service = feature_service

    async def predict(
        self,
        request: PredictionRequest,
        model_name: str = "ensemble",
    ) -> PredictionResponse:
        """
        Make prediction for a single post.

        Args:
            request: Prediction request
            model_name: Model to use for prediction

        Returns:
            Prediction response with metadata

        Raises:
            ValueError: If model not found or prediction fails
        """
        try:
            # Extract features
            features = await self.feature_service.extract_features(request)
            features_array = np.array(features).reshape(1, -1)

            # Load model
            model = self.model_loader.load_model(model_name)

            # Make prediction
            prediction = model.predict(features_array)[0]

            # Calculate confidence (using model's predict_proba if available)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_array)[0]
                confidence = float(np.max(proba))
            else:
                # Fallback: Use prediction magnitude as confidence proxy
                confidence = min(abs(prediction) / 10.0, 1.0)

            # Generate breakdown (simplified - in production, use separate models)
            breakdown = self._generate_breakdown(prediction, request.platform)

            # Create metadata
            metadata = PredictionMetadata(
                model_version=model_name,
                inference_time_ms=0.0,  # Will be set by router
                features_used=len(features),
                cached=False,
            )

            return PredictionResponse(
                engagement_score=float(prediction),
                confidence=float(confidence),
                breakdown=breakdown,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")

    def _generate_breakdown(
        self, total_score: float, platform: str
    ) -> EngagementScores:
        """
        Generate engagement breakdown.

        This is a simplified version. In production, you would use
        separate models for each metric.

        Args:
            total_score: Total engagement score
            platform: Social media platform

        Returns:
            Engagement breakdown
        """
        # Platform-specific distributions
        distributions = {
            "linkedin": {"likes": 0.6, "comments": 0.25, "shares": 0.15},
            "twitter": {"likes": 0.7, "comments": 0.15, "shares": 0.15},
            "bluesky": {"likes": 0.65, "comments": 0.20, "shares": 0.15},
        }

        dist = distributions.get(platform, distributions["linkedin"])

        return EngagementScores(
            likes=float(total_score * dist["likes"]),
            comments=float(total_score * dist["comments"]),
            shares=float(total_score * dist["shares"]),
        )