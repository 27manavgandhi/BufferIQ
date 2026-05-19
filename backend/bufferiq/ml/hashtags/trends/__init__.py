"""Hashtag trend detection."""

from bufferiq.ml.hashtags.trends.detector import (
    TrendDetector,
    TrendingHashtag,
    TrendStage,
)
from bufferiq.ml.hashtags.trends.momentum_scorer import MomentumScorer
from bufferiq.ml.hashtags.trends.viral_analyzer import ViralAnalyzer
from bufferiq.ml.hashtags.trends.realtime_monitor import RealtimeMonitor

__all__ = [
    "TrendDetector",
    "TrendingHashtag",
    "TrendStage",
    "MomentumScorer",
    "ViralAnalyzer",
    "RealtimeMonitor",
]