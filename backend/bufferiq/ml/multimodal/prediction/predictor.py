"""Main multi-modal predictor."""

from typing import Dict, Any, Optional
import numpy as np

from bufferiq.ml.multimodal.types import (
    EngagementPrediction,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import (
    PredictionError,
    UnsupportedPlatformError,
)
from bufferiq.ml.multimodal.prediction.impact_modeler import ImpactModeler
from bufferiq.ml.multimodal.prediction.ensemble import EnsemblePredictor


class MultiModalPredictor:
    """Predict engagement for multi-modal content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-modal predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.impact_modeler = ImpactModeler()
        self.ensemble = EnsemblePredictor()
        
        # Add impact modeler to ensemble
        self.ensemble.add_model(self.impact_modeler, weight=1.0)
    
    async def predict(
        self,
        features: np.ndarray,
        platform: PlatformType,
        baseline_engagement: float = 0.05
    ) -> EngagementPrediction:
        """
        Predict engagement for multi-modal content.
        
        Args:
            features: Multi-modal feature vector
            platform: Platform type
            baseline_engagement: Baseline engagement rate
            
        Returns:
            Engagement prediction
            
        Raises:
            UnsupportedPlatformError: If platform not supported
            PredictionError: If prediction fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        try:
            # Get prediction with confidence
            prediction, confidence = self.ensemble.predict_with_confidence(
                features,
                platform
            )
            
            # Calculate confidence interval
            margin = (1 - confidence) * prediction * 0.5
            confidence_interval = (
                max(0.0, prediction - margin),
                min(1.0, prediction + margin)
            )
            
            # Calculate improvement potential
            improvement = ((prediction - baseline_engagement) / baseline_engagement) * 100
            improvement = max(0.0, improvement)
            
            # Determine priority
            if improvement >= 30:
                priority = "high"
            elif improvement >= 15:
                priority = "medium"
            else:
                priority = "low"
            
            return EngagementPrediction(
                predicted_engagement_rate=prediction,
                confidence_interval=confidence_interval,
                improvement_potential=improvement,
                recommendation_priority=priority,
            )
            
        except Exception as e:
            if isinstance(e, UnsupportedPlatformError):
                raise
            raise PredictionError(f"Prediction failed: {str(e)}")