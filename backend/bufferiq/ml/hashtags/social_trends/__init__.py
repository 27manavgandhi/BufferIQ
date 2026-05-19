"""Social trends analysis."""

from bufferiq.ml.hashtags.social_trends.aggregator import TrendAggregator
from bufferiq.ml.hashtags.social_trends.viral_detector import ViralContentDetector
from bufferiq.ml.hashtags.social_trends.cultural_analyzer import CulturalAnalyzer

__all__ = [
    "TrendAggregator",
    "ViralContentDetector",
    "CulturalAnalyzer",
]