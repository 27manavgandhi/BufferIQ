"""Feature builder for multi-modal analysis."""

from typing import Dict, Any, Optional
import numpy as np

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    MultiModalFeatures,
    PlatformType,
)
from bufferiq.ml.multimodal.features.extractor import MultiModalFeatureExtractor
from bufferiq.ml.multimodal.features.fusion import FeatureFusion
from bufferiq.ml.multimodal.features.validator import FeatureValidator


class FeatureBuilder:
    """Build feature vectors for multi-modal content."""
    
    def __init__(self):
        """Initialize feature builder."""
        self.extractor = MultiModalFeatureExtractor()
        self.fusion = FeatureFusion(strategy="concatenate")
        self.validator = FeatureValidator()
    
    def build(
        self,
        text_features: Optional[Dict[str, Any]] = None,
        image_features: Optional[ImageAnalysisResult] = None,
        video_features: Optional[VideoAnalysisResult] = None,
        link_features: Optional[LinkPreviewAnalysis] = None,
        platform: PlatformType = "linkedin"
    ) -> MultiModalFeatures:
        """
        Build multi-modal feature vector.
        
        Args:
            text_features: Text analysis features
            image_features: Image analysis features
            video_features: Video analysis features
            link_features: Link preview features
            platform: Platform type
            
        Returns:
            Multi-modal features
        """
        # Extract features
        feature_vector = self.extractor.extract_features(
            text_features=text_features,
            image_features=image_features,
            video_features=video_features,
            link_features=link_features,
            platform=platform,
        )
        
        # Validate features
        is_valid, error_msg = self.validator.validate(feature_vector)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {error_msg}")
        
        # Normalize features
        normalized = self.fusion.normalize(feature_vector)
        
        # Get feature names
        feature_names = self.extractor.get_feature_names()
        
        # Determine modalities present
        modalities = []
        if text_features:
            modalities.append("text")
        if image_features:
            modalities.append("image")
        if video_features:
            modalities.append("video")
        if link_features:
            modalities.append("link")
        
        return MultiModalFeatures(
            feature_vector=normalized,
            feature_names=feature_names,
            modalities_present=modalities,
        )