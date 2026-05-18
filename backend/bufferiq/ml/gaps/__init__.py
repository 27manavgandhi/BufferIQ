"""
Content Gap Analysis & Competitive Intelligence Engine.

This module provides comprehensive content gap analysis, competitor benchmarking,
trend detection, and content recommendation capabilities.

Modules:
    - topics: Topic extraction and clustering
    - coverage: Content coverage analysis
    - detection: Gap detection and classification
    - competitors: Competitor analysis
    - trends: Trend detection
    - serp: Search engine analysis
    - recommendations: Content recommendations
    - calendar: Content calendar generation
    - benchmarks: Performance benchmarking
    - scoring: Opportunity scoring
    - intelligence: Main orchestrator service
"""

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService
from bufferiq.ml.gaps.topics.extractor import TopicExtractor, Topic, TopicCluster
from bufferiq.ml.gaps.coverage.mapper import CoverageMapper, CoverageMap
from bufferiq.ml.gaps.detection.detector import GapDetector, ContentGap, GapAnalysis
from bufferiq.ml.gaps.competitors.analyzer import (
    CompetitorAnalyzer,
    CompetitorProfile,
    CompetitiveAnalysis,
)
from bufferiq.ml.gaps.recommendations.generator import (
    ContentRecommendationEngine,
    ContentRecommendation,
)
from bufferiq.ml.gaps.calendar.generator import CalendarGenerator, ContentCalendar

__version__ = "1.0.0"

__all__ = [
    "GapIntelligenceService",
    "TopicExtractor",
    "Topic",
    "TopicCluster",
    "CoverageMapper",
    "CoverageMap",
    "GapDetector",
    "ContentGap",
    "GapAnalysis",
    "CompetitorAnalyzer",
    "CompetitorProfile",
    "CompetitiveAnalysis",
    "ContentRecommendationEngine",
    "ContentRecommendation",
    "CalendarGenerator",
    "ContentCalendar",
]