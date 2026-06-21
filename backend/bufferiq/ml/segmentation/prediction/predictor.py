"""Main engagement predictor for segments."""

from typing import Any, Dict

import numpy as np
from datetime import datetime

from bufferiq.ml.segmentation.types import SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import UnsupportedPlatformError
from bufferiq.ml.segmentation.prediction.segment_model import SegmentEngagementModel
from bufferiq.ml.segmentation.prediction.cross_segment import CrossSegmentAnalyzer
from bufferiq.ml.segmentation.prediction.calibrator import ModelCalibrator


class SegmentEngagementPredictor:
    """
    Predict engagement for audience segments.

    Includes:
    - Segment-specific models
    - Cross-segment analysis
    - Confidence intervals
    - Model calibration
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize predictor."""
        self.config = config or {}
        self.segment_models: Dict[str, SegmentEngagementModel] = {}
        self.cross_analyzer = CrossSegmentAnalyzer(self.config.get("cross_segment", {}))
        self.calibrator = ModelCalibrator(self.config.get("calibrator", {}))

    def fit_segment_model(
        self,
        segment_id: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fit engagement model for a segment.

        Args:
            segment_id: Segment identifier
            X: Feature matrix
            y: Engagement targets
        """
        model = SegmentEngagementModel(segment_id, self.config.get("model", {}))
        model.fit(X, y)
        self.segment_models[segment_id] = model

    def predict(
        self,
        segment_id: str,
        features: np.ndarray,
        baseline_engagement: float = 0.5,
        platform: str = "linkedin",
    ) -> Dict[str, Any]:
        """
        Predict engagement for a segment.

        Args:
            segment_id: Segment identifier
            features: Feature vector or matrix
            baseline_engagement: Baseline engagement for comparison
            platform: Platform type

        Returns:
            Engagement prediction with confidence

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if segment_id not in self.segment_models:
            return self._fallback_predict(baseline_engagement, platform)

        # Predict with segment model
        model = self.segment_models[segment_id]
        raw_prediction = model.predict(features.reshape(1, -1))[0]

        # Calibrate if calibrator is fitted
        if self.calibrator.is_fitted:
            calibrated = self.calibrator.calibrate(np.array([raw_prediction]))[0]
        else:
            calibrated = raw_prediction

        # Get confidence interval
        lower, upper = self.calibrator.get_confidence_interval(calibrated)

        # Calculate improvement
        improvement = (calibrated - baseline_engagement) / max(baseline_engagement, 0.1)

        # Determine priority
        priority = self._determine_priority(improvement, calibrated)

        return {
            "segment_id": segment_id,
            "predicted_engagement_rate": float(calibrated),
            "confidence_interval": (float(lower), float(upper)),
            "baseline_engagement": float(baseline_engagement),
            "improvement_potential": float(improvement),
            "recommendation_priority": priority,
            "platform": platform,
            "prediction_timestamp": datetime.utcnow().isoformat(),
        }

    def _determine_priority(self, improvement: float, engagement: float) -> str:
        """Determine recommendation priority."""
        if improvement > 0.2 or engagement > 0.8:
            return "high"
        elif improvement > 0.1 or engagement > 0.5:
            return "medium"
        else:
            return "low"

    def _fallback_predict(
        self, baseline_engagement: float, platform: str
    ) -> Dict[str, Any]:
        """Fallback prediction without segment model."""
        # Platform adjustments
        platform_multiplier = {
            "linkedin": 1.1,
            "twitter": 1.0,
            "bluesky": 0.95,
        }.get(platform, 1.0)

        predicted = baseline_engagement * platform_multiplier
        predicted = min(max(predicted, 0.0), 1.0)

        return {
            "predicted_engagement_rate": float(predicted),
            "confidence_interval": (float(predicted - 0.1), float(predicted + 0.1)),
            "baseline_engagement": float(baseline_engagement),
            "improvement_potential": float(predicted - baseline_engagement),
            "recommendation_priority": "low",
            "platform": platform,
            "prediction_timestamp": datetime.utcnow().isoformat(),
        }