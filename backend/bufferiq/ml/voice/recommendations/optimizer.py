"""
Voice optimization engine.

Optimizes content for maximum voice alignment.
"""

from typing import Optional
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile
from bufferiq.ml.voice.recommendations.generator import (
    VoiceOptimizationResult,
    VoiceRecommendationEngine,
)
from bufferiq.ml.voice.recommendations.rewriter import VoiceRewriter
from bufferiq.ml.voice.consistency.scorer import VoiceConsistencyScorer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class VoiceOptimizer:
    """
    Optimize content for voice alignment.
    
    Combines analysis, recommendations, and rewriting
    to maximize voice consistency.
    
    Example:
```python
        optimizer = VoiceOptimizer()
        result = optimizer.optimize(
            text="Your content here",
            profile=brand_voice,
            platform="linkedin"
        )
        print(f"Improved by {result.improvement:.1f} points")
```
    """
    
    def __init__(
        self,
        scorer: Optional[VoiceConsistencyScorer] = None,
        engine: Optional[VoiceRecommendationEngine] = None,
        rewriter: Optional[VoiceRewriter] = None,
    ):
        """Initialize voice optimizer."""
        self.scorer = scorer or VoiceConsistencyScorer()
        self.engine = engine or VoiceRecommendationEngine()
        self.rewriter = rewriter or VoiceRewriter()
    
    def optimize(
        self,
        text: str,
        profile: VoiceProfile,
        platform: str,
        max_iterations: int = 3,
    ) -> VoiceOptimizationResult:
        """
        Optimize content for voice alignment.
        
        Args:
            text: Content to optimize
            profile: Voice profile
            platform: Target platform
            max_iterations: Maximum optimization iterations
        
        Returns:
            Optimization result
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        logger.info(f"Starting optimization for {platform}")
        
        # Use the recommendation engine's optimize method
        result = self.engine.optimize(text, profile, platform)
        
        logger.info(
            f"Optimization complete: {result.original_score:.1f} -> "
            f"{result.optimized_score:.1f} (+{result.improvement:.1f})"
        )
        
        return result