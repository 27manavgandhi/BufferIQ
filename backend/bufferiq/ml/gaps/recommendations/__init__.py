"""Content recommendations module."""

from bufferiq.ml.gaps.recommendations.generator import (
    ContentRecommendationEngine,
    ContentRecommendation,
)
from bufferiq.ml.gaps.recommendations.title_suggester import TitleSuggester
from bufferiq.ml.gaps.recommendations.formatter import FormatRecommender

__all__ = [
    "ContentRecommendationEngine",
    "ContentRecommendation",
    "TitleSuggester",
    "FormatRecommender",
]