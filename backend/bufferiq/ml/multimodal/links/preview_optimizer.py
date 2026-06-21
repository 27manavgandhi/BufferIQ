"""Link preview optimization."""

from typing import List, Dict
from bufferiq.ml.multimodal.types import LinkMetadata


class PreviewOptimizer:
    """Optimize link previews for engagement."""
    
    def __init__(self):
        """Initialize preview optimizer."""
        pass
    
    def optimize_title(self, title: str, platform: str) -> str:
        """
        Optimize title for platform.
        
        Args:
            title: Original title
            platform: Platform type
            
        Returns:
            Optimized title
        """
        if not title:
            return title
        
        # Platform-specific title length limits
        max_lengths = {
            "linkedin": 70,
            "twitter": 50,
            "bluesky": 60,
        }
        
        max_length = max_lengths.get(platform, 70)
        
        if len(title) > max_length:
            # Truncate with ellipsis
            return title[:max_length-3] + "..."
        
        return title
    
    def optimize_description(self, description: str, platform: str) -> str:
        """
        Optimize description for platform.
        
        Args:
            description: Original description
            platform: Platform type
            
        Returns:
            Optimized description
        """
        if not description:
            return description
        
        # Platform-specific description length limits
        max_lengths = {
            "linkedin": 200,
            "twitter": 120,
            "bluesky": 150,
        }
        
        max_length = max_lengths.get(platform, 200)
        
        if len(description) > max_length:
            # Truncate at last complete sentence or word
            truncated = description[:max_length]
            
            # Try to end at sentence
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # At least 70% of max length
                return truncated[:last_period+1]
            
            # Otherwise end at last word
            last_space = truncated.rfind(' ')
            if last_space > 0:
                return truncated[:last_space] + "..."
            
            return truncated + "..."
        
        return description
    
    def generate_suggestions(
        self,
        metadata: LinkMetadata,
        quality_scores: Dict[str, float],
        platform: str
    ) -> List[str]:
        """
        Generate optimization suggestions.
        
        Args:
            metadata: Link metadata
            quality_scores: Quality scores
            platform: Platform type
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Title suggestions
        if quality_scores.get('title_quality', 0) < 70:
            if not metadata.title:
                suggestions.append("Add a compelling title")
            elif len(metadata.title) < 30:
                suggestions.append("Expand title to be more descriptive")
            elif len(metadata.title) > 70:
                suggestions.append("Shorten title for better readability")
        
        # Description suggestions
        if quality_scores.get('description_quality', 0) < 70:
            if not metadata.description:
                suggestions.append("Add a description")
            elif len(metadata.description) < 50:
                suggestions.append("Expand description with more details")
        
        # Image suggestions
        if quality_scores.get('image_quality', 0) < 70:
            if not metadata.image_url:
                suggestions.append("Add a preview image")
            else:
                suggestions.append("Use a higher quality preview image")
        
        # Open Graph tags
        if not metadata.og_tags:
            suggestions.append("Add Open Graph meta tags")
        
        # Twitter Card tags (for Twitter/Bluesky)
        if platform in ["twitter", "bluesky"] and not metadata.twitter_tags:
            suggestions.append("Add Twitter Card meta tags")
        
        return suggestions