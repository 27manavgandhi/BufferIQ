"""Main link preview analyzer orchestrating all components."""

from typing import Dict, Any, Optional
import time

from bufferiq.ml.multimodal.types import (
    LinkPreviewAnalysis,
    QualityScores,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError
from bufferiq.ml.multimodal.links.metadata_extractor import LinkMetadataExtractor
from bufferiq.ml.multimodal.links.preview_optimizer import PreviewOptimizer
from bufferiq.ml.multimodal.links.quality_scorer import QualityScorer
from bufferiq.ml.multimodal.links.ctr_predictor import CTRPredictor


class LinkPreviewAnalyzer:
    """Comprehensive link preview analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize link preview analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.metadata_extractor = LinkMetadataExtractor()
        self.optimizer = PreviewOptimizer()
        self.quality_scorer = QualityScorer()
        self.ctr_predictor = CTRPredictor()
    
    async def analyze(
        self,
        url: str,
        platform: PlatformType
    ) -> LinkPreviewAnalysis:
        """
        Analyze link preview.
        
        Args:
            url: URL to analyze
            platform: Platform type (linkedin/twitter/bluesky)
            
        Returns:
            Link preview analysis result
            
        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        # Validate platform
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        start_time = time.time()
        
        # Extract metadata
        metadata = await self.metadata_extractor.extract(url)
        
        # Score quality
        title_quality = self.quality_scorer.score_title(metadata.title)
        description_quality = self.quality_scorer.score_description(metadata.description)
        image_quality = self.quality_scorer.score_image(metadata.image_url)
        overall_quality = self.quality_scorer.score_overall(
            title_quality,
            description_quality,
            image_quality
        )
        
        quality_scores = QualityScores(
            title_quality=title_quality,
            description_quality=description_quality,
            image_quality=image_quality,
            overall_quality=overall_quality,
        )
        
        # Predict CTR
        ctr = self.ctr_predictor.predict(
            metadata,
            quality_scores.to_dict(),
            platform
        )
        
        # Generate suggestions
        suggestions = self.optimizer.generate_suggestions(
            metadata,
            quality_scores.to_dict(),
            platform
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return LinkPreviewAnalysis(
            metadata=metadata,
            quality_scores=quality_scores,
            ctr_prediction=ctr,
            optimization_suggestions=suggestions,
            platform=platform,
            processing_time_ms=processing_time_ms,
        )