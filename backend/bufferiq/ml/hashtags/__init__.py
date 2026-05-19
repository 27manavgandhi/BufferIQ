"""
Hashtag Optimizer & Social Trends Intelligence System.

This module provides comprehensive hashtag analysis, including:
- Hashtag extraction and normalization
- Performance analysis and ROI calculation
- Trend detection and momentum scoring
- Related hashtag discovery
- Platform-specific strategies
- Effectiveness prediction
- Risk and safety detection
- Combination optimization
- Social trends aggregation
- Lifecycle tracking
"""

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService
from bufferiq.ml.hashtags.extraction.extractor import (
    HashtagExtractor,
    ExtractedHashtag,
    HashtagExtractionResult,
)
from bufferiq.ml.hashtags.performance.analyzer import (
    HashtagPerformanceAnalyzer,
    HashtagPerformance,
)
from bufferiq.ml.hashtags.trends.detector import (
    TrendDetector,
    TrendingHashtag,
    TrendStage,
)
from bufferiq.ml.hashtags.discovery.engine import (
    HashtagDiscoveryEngine,
    HashtagDiscovery,
    RelatedHashtag,
)
from bufferiq.ml.hashtags.strategy.generator import (
    HashtagStrategyGenerator,
    HashtagStrategy,
)
from bufferiq.ml.hashtags.risks.detector import (
    HashtagRiskDetector,
    HashtagRisk,
)

__all__ = [
    # Main service
    "HashtagIntelligenceService",
    # Extraction
    "HashtagExtractor",
    "ExtractedHashtag",
    "HashtagExtractionResult",
    # Performance
    "HashtagPerformanceAnalyzer",
    "HashtagPerformance",
    # Trends
    "TrendDetector",
    "TrendingHashtag",
    "TrendStage",
    # Discovery
    "HashtagDiscoveryEngine",
    "HashtagDiscovery",
    "RelatedHashtag",
    # Strategy
    "HashtagStrategyGenerator",
    "HashtagStrategy",
    # Risks
    "HashtagRiskDetector",
    "HashtagRisk",
]

__version__ = "1.0.0"