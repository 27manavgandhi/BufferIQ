"""Generate optimization recommendations."""

from typing import List, Dict, Any, Optional
import numpy as np

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
)


class OptimizationRecommender:
    """Generate recommendations for multi-modal content optimization."""
    
    def __init__(self):
        """Initialize optimization recommender."""
        pass
    
    def generate_recommendations(
        self,
        text: str,
        image_results: List[ImageAnalysisResult],
        video_results: List[VideoAnalysisResult],
        link_results: List[LinkPreviewAnalysis],
        platform: str
    ) -> List[str]:
        """
        Generate optimization recommendations.
        
        Args:
            text: Post text
            image_results: Image analysis results
            video_results: Video analysis results
            link_results: Link preview results
            platform: Platform type
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Text recommendations
        recommendations.extend(self._text_recommendations(text, platform))
        
        # Image recommendations
        for img in image_results:
            recommendations.extend(self._image_recommendations(img, platform))
        
        # Video recommendations
        for vid in video_results:
            recommendations.extend(self._video_recommendations(vid, platform))
        
        # Link recommendations
        for link in link_results:
            recommendations.extend(link.optimization_suggestions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _text_recommendations(self, text: str, platform: str) -> List[str]:
        """Generate text-based recommendations."""
        recommendations = []
        
        # Length recommendations
        word_count = len(text.split())
        
        optimal_lengths = {
            "linkedin": (100, 300),
            "twitter": (50, 150),
            "bluesky": (50, 200),
        }
        
        min_words, max_words = optimal_lengths.get(platform, (50, 300))
        
        if word_count < min_words:
            recommendations.append(
                f"Expand text to at least {min_words} words for better engagement"
            )
        elif word_count > max_words:
            recommendations.append(
                f"Shorten text to under {max_words} words for optimal engagement"
            )
        
        # Hashtag recommendations
        hashtag_count = text.count('#')
        if hashtag_count == 0:
            recommendations.append("Add 2-3 relevant hashtags to increase discoverability")
        elif hashtag_count > 5:
            recommendations.append("Reduce number of hashtags to 3-5 for better readability")
        
        return recommendations
    
    def _image_recommendations(
        self,
        image: ImageAnalysisResult,
        platform: str
    ) -> List[str]:
        """Generate image-based recommendations."""
        recommendations = []
        
        # Aesthetic score
        if image.aesthetic_score < 60:
            recommendations.append("Use a higher quality image with better composition")
        
        # Face presence (generally increases engagement)
        if not image.faces and platform == "linkedin":
            recommendations.append(
                "Consider using an image with people for higher engagement on LinkedIn"
            )
        
        # Text in image
        if len(image.text) > 3:
            recommendations.append(
                "Reduce amount of text in image - visual content should be primary"
            )
        
        # Brand elements
        if not image.brand_elements and platform == "linkedin":
            recommendations.append(
                "Add subtle brand watermark to increase brand awareness"
            )
        
        # Composition
        if image.composition.rule_of_thirds < 0.4:
            recommendations.append(
                "Improve image composition using the rule of thirds"
            )
        
        return recommendations
    
    def _video_recommendations(
        self,
        video: VideoAnalysisResult,
        platform: str
    ) -> List[str]:
        """Generate video-based recommendations."""
        recommendations = []
        
        # Duration
        optimal_durations = {
            "linkedin": (30, 90),
            "twitter": (15, 60),
            "bluesky": (15, 60),
        }
        
        min_dur, max_dur = optimal_durations.get(platform, (15, 90))
        
        if video.metadata.duration_seconds < min_dur:
            recommendations.append(
                f"Extend video to at least {min_dur} seconds for better storytelling"
            )
        elif video.metadata.duration_seconds > max_dur:
            recommendations.append(
                f"Shorten video to under {max_dur} seconds for optimal engagement"
            )
        
        # Resolution
        if video.metadata.resolution[0] < 1280:
            recommendations.append("Use HD resolution (1280x720 or higher) for better quality")
        
        # Audio
        if not video.metadata.has_audio:
            recommendations.append(
                "Consider adding background music or narration to enhance engagement"
            )
        
        # Scenes
        if len(video.scenes) < 3 and video.metadata.duration_seconds > 30:
            recommendations.append(
                "Add more scene transitions to maintain viewer interest"
            )
        
        return recommendations