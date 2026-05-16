"""
Voice Profile Analyzer & Brand Consistency Engine.

This module provides comprehensive voice analysis and brand consistency
monitoring for social media content.

Modules:
    - linguistic: Lexical and syntactic analysis
    - stylistic: Writing style and tone detection
    - extraction: Voice feature extraction from historical content
    - profiler: Voice profile building and management
    - consistency: Voice consistency scoring
    - drift: Voice drift detection
    - recommendations: Voice-aligned content suggestions
    - brands: Multi-brand voice management
    - validation: Pre-publish validation
    - intelligence: Main orchestrator service

Example:
```python
    from bufferiq.ml.voice import VoiceIntelligenceService
    
    service = VoiceIntelligenceService(db_session=session)
    profile = await service.build_voice_profile(
        brand_id="brand123",
        platform="linkedin"
    )
```
"""

from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService
from bufferiq.ml.voice.profiler.builder import VoiceProfile, VoiceProfileBuilder
from bufferiq.ml.voice.consistency.scorer import (
    ConsistencyScore,
    VoiceConsistencyScorer,
)
from bufferiq.ml.voice.drift.detector import DriftAlert, VoiceDriftDetector
from bufferiq.ml.voice.linguistic.lexical_analyzer import (
    LexicalAnalyzer,
    LexicalMetrics,
)
from bufferiq.ml.voice.linguistic.syntactic_analyzer import (
    SyntacticAnalyzer,
    SyntacticMetrics,
)
from bufferiq.ml.voice.stylistic.style_detector import (
    StyleDetector,
    StylisticFeatures,
    WritingStyle,
)
from bufferiq.ml.voice.extraction.extractor import VoiceExtractor, VoiceFeatures
from bufferiq.ml.voice.recommendations.generator import (
    VoiceRecommendation,
    VoiceRecommendationEngine,
    VoiceOptimizationResult,
)

__version__ = "1.0.0"

__all__ = [
    # Main service
    "VoiceIntelligenceService",
    # Profile management
    "VoiceProfile",
    "VoiceProfileBuilder",
    # Consistency
    "ConsistencyScore",
    "VoiceConsistencyScorer",
    # Drift detection
    "DriftAlert",
    "VoiceDriftDetector",
    # Linguistic analysis
    "LexicalAnalyzer",
    "LexicalMetrics",
    "SyntacticAnalyzer",
    "SyntacticMetrics",
    # Stylistic analysis
    "StyleDetector",
    "StylisticFeatures",
    "WritingStyle",
    # Voice extraction
    "VoiceExtractor",
    "VoiceFeatures",
    # Recommendations
    "VoiceRecommendation",
    "VoiceRecommendationEngine",
    "VoiceOptimizationResult",
]

# Platform constants
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]

def validate_platform(platform: str) -> None:
    """
    Validate platform is supported.
    
    Args:
        platform: Platform to validate
        
    Raises:
        ValueError: If platform not supported
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' not supported. "
            f"Supported platforms: {SUPPORTED_PLATFORMS}"
        )