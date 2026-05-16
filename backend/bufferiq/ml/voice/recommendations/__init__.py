"""
Voice recommendations module.

Generates voice-aligned content suggestions and rewrites.
"""

from bufferiq.ml.voice.recommendations.generator import (
    VoiceRecommendation,
    VoiceRecommendationEngine,
)
from bufferiq.ml.voice.recommendations.rewriter import VoiceRewriter
from bufferiq.ml.voice.recommendations.optimizer import (
    VoiceOptimizationResult,
    VoiceOptimizer,
)

__all__ = [
    "VoiceRecommendation",
    "VoiceRecommendationEngine",
    "VoiceRewriter",
    "VoiceOptimizationResult",
    "VoiceOptimizer",
]