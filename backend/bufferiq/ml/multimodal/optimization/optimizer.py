"""Main multi-modal optimizer."""

from typing import Dict, List, Any, Optional

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError
from bufferiq.ml.multimodal.optimization.recommender import OptimizationRecommender
from bufferiq.ml.multimodal.optimization.ab_integration import ABTestingIntegration


class MultiModalOptimizer:
    """Optimize multi-modal content for engagement."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-modal optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.recommender = OptimizationRecommender()
        self.ab_testing = ABTestingIntegration()
    
    async def optimize(
        self,
        text: str,
        image_results: List[ImageAnalysisResult],
        video_results: List[VideoAnalysisResult],
        link_results: List[LinkPreviewAnalysis],
        platform: PlatformType
    ) -> Dict[str, Any]:
        """
        Optimize multi-modal content.
        
        Args:
            text: Post text
            image_results: Image analysis results
            video_results: Video analysis results
            link_results: Link preview results
            platform: Platform type
            
        Returns:
            Optimization results
            
        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        # Generate recommendations
        recommendations = self.recommender.generate_recommendations(
            text,
            image_results,
            video_results,
            link_results,
            platform
        )
        
        # Prioritize recommendations
        prioritized = self._prioritize_recommendations(recommendations)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            image_results,
            video_results,
            link_results
        )
        
        return {
            "recommendations": prioritized,
            "optimization_score": optimization_score,
            "total_recommendations": len(recommendations),
            "high_priority_count": len([r for r in prioritized if r["priority"] == "high"]),
            "platform": platform,
        }
    
    def _prioritize_recommendations(
        self,
        recommendations: List[str]
    ) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact."""
        prioritized = []
        
        # Keywords for high priority
        high_priority_keywords = [
            "quality",
            "resolution",
            "engagement",
            "add hashtags",
        ]
        
        # Keywords for medium priority
        medium_priority_keywords = [
            "shorten",
            "expand",
            "improve",
            "consider",
        ]
        
        for rec in recommendations:
            # Determine priority
            rec_lower = rec.lower()
            
            if any(keyword in rec_lower for keyword in high_priority_keywords):
                priority = "high"
            elif any(keyword in rec_lower for keyword in medium_priority_keywords):
                priority = "medium"
            else:
                priority = "low"
            
            prioritized.append({
                "recommendation": rec,
                "priority": priority,
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        prioritized.sort(key=lambda x: priority_order[x["priority"]])
        
        return prioritized
    
    def _calculate_optimization_score(
        self,
        image_results: List[ImageAnalysisResult],
        video_results: List[VideoAnalysisResult],
        link_results: List[LinkPreviewAnalysis]
    ) -> float:
        """
        Calculate overall optimization score.
        
        Args:
            image_results: Image analysis results
            video_results: Video analysis results
            link_results: Link preview results
            
        Returns:
            Optimization score (0-100)
        """
        scores = []
        
        # Image scores
        for img in image_results:
            scores.append(img.aesthetic_score)
        
        # Video scores (based on engagement prediction)
        for vid in video_results:
            scores.append(vid.engagement_prediction * 100)
        
        # Link scores
        for link in link_results:
            scores.append(link.quality_scores.overall_quality)
        
        if not scores:
            return 50.0  # Neutral score
        
        return sum(scores) / len(scores)