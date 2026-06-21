"""Ensemble prediction combining multiple models."""

from typing import List, Dict, Any
import numpy as np


class EnsemblePredictor:
    """Ensemble predictor for multi-modal engagement."""
    
    def __init__(self, models: List[Any] | None = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of prediction models
        """
        self.models = models or []
        self.weights: List[float] = []
    
    def add_model(self, model: Any, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Prediction model
            weight: Model weight
        """
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(
        self,
        features: np.ndarray,
        platform: str
    ) -> float:
        """
        Predict using ensemble.
        
        Args:
            features: Feature vector
            platform: Platform type
            
        Returns:
            Ensemble prediction
        """
        if not self.models:
            # Fallback to simple heuristic
            return self._fallback_predict(features, platform)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict_impact(features, platform)
            predictions.append(pred)
        
        # Weighted average
        weights_array = np.array(self.weights)
        weights_array = weights_array / weights_array.sum()
        
        ensemble_pred = np.average(predictions, weights=weights_array)
        
        return float(ensemble_pred)
    
    def _fallback_predict(self, features: np.ndarray, platform: str) -> float:
        """Fallback prediction when no models available."""
        base_score = 0.5
        
        # Platform adjustment
        platform_scores = {
            "linkedin": 0.55,
            "twitter": 0.50,
            "bluesky": 0.48,
        }
        
        return platform_scores.get(platform, base_score)
    
    def predict_with_confidence(
        self,
        features: np.ndarray,
        platform: str
    ) -> tuple[float, float]:
        """
        Predict with confidence interval.
        
        Args:
            features: Feature vector
            platform: Platform type
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.models:
            return self._fallback_predict(features, platform), 0.5
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict_impact(features, platform)
            predictions.append(pred)
        
        # Calculate mean and std
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence is inverse of std (lower std = higher confidence)
        confidence = 1.0 - min(std_pred * 2, 1.0)
        
        return float(mean_pred), float(confidence)