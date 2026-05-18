"""Topic extraction and analysis module."""

from bufferiq.ml.gaps.topics.extractor import TopicExtractor, Topic, TopicCluster
from bufferiq.ml.gaps.topics.clusterer import TopicClusterer
from bufferiq.ml.gaps.topics.trend_detector import TrendDetector, TopicTrend
from bufferiq.ml.gaps.topics.lifecycle_analyzer import (
    LifecycleAnalyzer,
    LifecycleStage,
)

__all__ = [
    "TopicExtractor",
    "Topic",
    "TopicCluster",
    "TopicClusterer",
    "TrendDetector",
    "TopicTrend",
    "LifecycleAnalyzer",
    "LifecycleStage",
]