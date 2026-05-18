"""Trend analysis module."""

from bufferiq.ml.gaps.trends.detector import TrendDetector, TrendSignal
from bufferiq.ml.gaps.trends.momentum_scorer import MomentumScorer
from bufferiq.ml.gaps.trends.seasonal_analyzer import SeasonalAnalyzer

__all__ = [
    "TrendDetector",
    "TrendSignal",
    "MomentumScorer",
    "SeasonalAnalyzer",
]