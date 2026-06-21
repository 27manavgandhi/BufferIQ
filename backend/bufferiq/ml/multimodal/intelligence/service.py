"""Unified multi-modal intelligence service."""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError
from bufferiq.ml.multimodal.images.analyzer import ImageAnalyzer
from bufferiq.ml.multimodal.videos.analyzer import VideoAnalyzer
from bufferiq.ml.multimodal.links.analyzer import LinkPreviewAnalyzer
from bufferiq.ml.multimodal.features.builder import FeatureBuilder
from bufferiq.ml.multimodal.prediction.predictor import MultiModalPredictor
from bufferiq.ml.multimodal.optimization.optimizer import MultiModalOptimizer


class MultiModalIntelligenceService:
    """Unified service for multi-modal content analysis and optimization."""
    
    def __init__(
        self,
        image_analyzer: Optional[ImageAnalyzer] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
        link_analyzer: Optional[LinkPreviewAnalyzer] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        predictor: Optional[MultiModalPredictor] = None,
        optimizer: Optional[MultiModalOptimizer] = None,
    ):
        """
        Initialize multi-modal intelligence service.
        
        Args:
            image_analyzer: Image analyzer instance
            video_analyzer: Video analyzer instance
            link_analyzer: Link preview analyzer instance
            feature_builder: Feature builder instance
            predictor: Multi-modal predictor instance
            optimizer: Multi-modal optimizer instance
        """
        self.image_analyzer = image_analyzer or ImageAnalyzer()
        self.video_analyzer = video_analyzer or VideoAnalyzer()
        self.link_analyzer = link_analyzer or LinkPreviewAnalyzer()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.predictor = predictor or MultiModalPredictor()
        self.optimizer = optimizer or MultiModalOptimizer()
    
    async def analyze_post(
        self,
        post_id: str,
        text: str,
        image_urls: Optional[List[str]] = None,
        video_urls: Optional[List[str]] = None,
        link_urls: Optional[List[str]] = None,
        platform: PlatformType = "linkedin"
    ) -> Dict[str, Any]:
        """
        Analyze post with all modalities.
        
        Args:
            post_id: Post identifier
            text: Post text
            image_urls: List of image URLs
            video_urls: List of video URLs
            link_urls: List of link URLs
            platform: Platform type (linkedin/twitter/bluesky)
            
        Returns:
            Comprehensive multi-modal analysis
            
        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        start_time = datetime.now()
        
        # Analyze all modalities in parallel
        tasks = []
        
        if image_urls:
            for url in image_urls:
                tasks.append(self.image_analyzer.analyze(url, platform))
        
        if video_urls:
            for url in video_urls:
                tasks.append(self.video_analyzer.analyze(url, platform))
        
        if link_urls:
            for url in link_urls:
                tasks.append(self.link_analyzer.analyze(url, platform))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate results by type
        image_results: List[ImageAnalysisResult] = []
        video_results: List[VideoAnalysisResult] = []
        link_results: List[LinkPreviewAnalysis] = []
        
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue
                continue
            elif isinstance(result, ImageAnalysisResult):
                image_results.append(result)
            elif isinstance(result, VideoAnalysisResult):
                video_results.append(result)
            elif isinstance(result, LinkPreviewAnalysis):
                link_results.append(result)
        
        # Build features
        text_features = {"text": text}
        
        multimodal_features = self.feature_builder.build(
            text_features=text_features,
            image_features=image_results[0] if image_results else None,
            video_features=video_results[0] if video_results else None,
            link_features=link_results[0] if link_results else None,
            platform=platform,
        )
        
        # Predict engagement
        engagement_prediction = await self.predictor.predict(
            multimodal_features.feature_vector,
            platform
        )
        
        # Generate optimizations
        optimization = await self.optimizer.optimize(
            text,
            image_results,
            video_results,
            link_results,
            platform
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "post_id": post_id,
            "platform": platform,
            "analysis": {
                "images": [self._serialize_image_result(r) for r in image_results],
                "videos": [self._serialize_video_result(r) for r in video_results],
                "links": [self._serialize_link_result(r) for r in link_results],
            },
            "features": {
                "dimension": len(multimodal_features.feature_vector),
                "modalities": multimodal_features.modalities_present,
            },
            "engagement_prediction": engagement_prediction.to_dict(),
            "optimization": optimization,
            "processing_time_ms": processing_time,
        }
    
    def _serialize_image_result(self, result: ImageAnalysisResult) -> Dict[str, Any]:
        """Serialize image analysis result."""
        return result.to_dict()
    
    def _serialize_video_result(self, result: VideoAnalysisResult) -> Dict[str, Any]:
        """Serialize video analysis result."""
        return result.to_dict()
    
    def _serialize_link_result(self, result: LinkPreviewAnalysis) -> Dict[str, Any]:
        """Serialize link preview result."""
        return result.to_dict()