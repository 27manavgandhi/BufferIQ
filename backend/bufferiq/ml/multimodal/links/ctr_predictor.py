"""CTR prediction for link previews."""

from typing import Dict
import numpy as np

from bufferiq.ml.multimodal.types import LinkMetadata


class CTRPredictor:
    """Predict click-through rate for link previews."""
    
    def __init__(self):
        """Initialize CTR predictor."""
        # In production, load trained model
        pass
    
    def predict(
        self,
        metadata: LinkMetadata,
        quality_scores: Dict[str, float],
        platform: str
    ) -> float:
        """
        Predict CTR for link preview.
        
        Args:
            metadata: Link metadata
            quality_scores: Quality scores
            platform: Platform type
            
        Returns:
            Predicted CTR (0-1)
        """
        # Feature engineering
        features = self._extract_features(metadata, quality_scores, platform)
        
        # Simple heuristic model (in production, use trained ML model)
        base_ctr = 0.02  # 2% base CTR
        
        # Quality multiplier
        quality_multiplier = quality_scores['overall_quality'] / 100.0
        
        # Platform adjustment
        platform_multipliers = {
            "linkedin": 1.2,  # Higher CTR on LinkedIn
            "twitter": 1.0,
            "bluesky": 0.9,
        }
        platform_multiplier = platform_multipliers.get(platform, 1.0)
        
        # Has image bonus
        image_bonus = 1.3 if metadata.image_url else 1.0
        
        # Calculate predicted CTR
        predicted_ctr = (
            base_ctr *
            quality_multiplier *
            platform_multiplier *
            image_bonus
        )
        
        return min(predicted_ctr, 0.15)  # Cap at 15%
    
    def _extract_features(
        self,
        metadata: LinkMetadata,
        quality_scores: Dict[str, float],
        platform: str
    ) -> np.ndarray:
        """
        Extract features for CTR prediction.
        
        Args:
            metadata: Link metadata
            quality_scores: Quality scores
            platform: Platform type
            
        Returns:
            Feature vector
        """
        features = []
        
        # Quality scores
        features.append(quality_scores.get('title_quality', 0) / 100.0)
        features.append(quality_scores.get('description_quality', 0) / 100.0)
        features.append(quality_scores.get('image_quality', 0) / 100.0)
        
        # Has image
        features.append(1.0 if metadata.image_url else 0.0)
        
        # Title length
        title_length = len(metadata.title) if metadata.title else 0
        features.append(min(title_length / 70.0, 1.0))
        
        # Description length
        desc_length = len(metadata.description) if metadata.description else 0
        features.append(min(desc_length / 200.0, 1.0))
        
        # Platform encoding (one-hot)
        platform_encoding = [0.0, 0.0, 0.0]
        platform_idx = {"linkedin": 0, "twitter": 1, "bluesky": 2}.get(platform, 0)
        platform_encoding[platform_idx] = 1.0
        features.extend(platform_encoding)
        
        return np.array(features)