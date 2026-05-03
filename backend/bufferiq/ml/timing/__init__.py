"""
Timing Intelligence Module for BufferIQ.

Provides comprehensive time-series analysis, forecasting, and timing optimization
for social media posting. Uses Prophet for forecasting, statistical pattern detection,
and multi-platform timing coordination.

Public API:
    - TimeSeriesBuilder: Build time-series from posts
    - ProphetForecaster: Forecast audience activity
    - EngagementWindowDetector: Detect high-engagement windows
    - OptimalTimeFinder: Find best posting times
    - TimingRecommender: Main orchestrator for timing recommendations
"""

from bufferiq.ml.timing.data_preparation.time_series_builder import TimeSeriesBuilder
from bufferiq.ml.timing.forecasting.prophet_forecaster import ProphetForecaster
from bufferiq.ml.timing.pattern_detection.engagement_window_detector import (
    EngagementWindowDetector,
    EngagementWindow,
)
from bufferiq.ml.timing.optimization.optimal_time_finder import (
    OptimalTimeFinder,
    TimingRecommendation,
)
from bufferiq.ml.timing.recommendation.timing_recommender import TimingRecommender

__all__ = [
    "TimeSeriesBuilder",
    "ProphetForecaster",
    "EngagementWindowDetector",
    "EngagementWindow",
    "OptimalTimeFinder",
    "TimingRecommendation",
    "TimingRecommender",
]