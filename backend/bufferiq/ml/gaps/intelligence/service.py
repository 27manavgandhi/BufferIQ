"""
Main gap intelligence orchestration service.

Coordinates all gap analysis modules and provides unified interface.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.gaps.topics.extractor import TopicExtractor
from bufferiq.ml.gaps.coverage.mapper import CoverageMapper
from bufferiq.ml.gaps.detection.detector import GapDetector
from bufferiq.ml.gaps.competitors.analyzer import CompetitorAnalyzer
from bufferiq.ml.gaps.recommendations.generator import ContentRecommendationEngine
from bufferiq.ml.gaps.calendar.generator import CalendarGenerator
from bufferiq.ml.gaps.benchmarks.tracker import BenchmarkTracker
from bufferiq.ml.gaps.scoring.scorer import OpportunityScorer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class GapIntelligenceService:
    """
    Main orchestrator for gap analysis and competitive intelligence.

    Coordinates all gap analysis modules:
    - Topic extraction
    - Coverage analysis
    - Gap detection
    - Competitor analysis
    - Trend detection
    - Recommendations
    - Calendar generation

    Example:
```python
        service = GapIntelligenceService(
            db_session=session,
            cache=redis_client
        )

        # Comprehensive gap analysis
        report = await service.analyze_gaps(
            user_id="user123",
            platform="linkedin",
            competitor_ids=["comp1", "comp2"],
            industry="technology",
            include_recommendations=True
        )

        print(f"Coverage: {report['coverage_score']:.1f}%")
        print(f"Gaps found: {report['total_gaps']}")
        print(f"Critical gaps: {len(report['critical_gaps'])}")
        print(f"Recommendations: {len(report['recommendations'])}")

        # Generate content calendar
        calendar = await service.generate_calendar(
            user_id="user123",
            platform="linkedin",
            weeks=4,
            posts_per_week=3
        )

        print(f"\\nCalendar: {calendar['total_pieces']} pieces over 4 weeks")
```
    """

    def __init__(
        self,
        db_session: Session,
        cache: Optional[Any] = None,
        topic_extractor: Optional[TopicExtractor] = None,
        coverage_mapper: Optional[CoverageMapper] = None,
        gap_detector: Optional[GapDetector] = None,
        competitor_analyzer: Optional[CompetitorAnalyzer] = None,
        recommendation_engine: Optional[ContentRecommendationEngine] = None,
        calendar_generator: Optional[CalendarGenerator] = None,
        benchmark_tracker: Optional[BenchmarkTracker] = None,
        opportunity_scorer: Optional[OpportunityScorer] = None,
    ):
        """Initialize gap intelligence service."""
        self.db = db_session
        self.cache = cache

        # Initialize components
        self.topic_extractor = topic_extractor or TopicExtractor(db_session)
        self.coverage_mapper = coverage_mapper or CoverageMapper(db_session)
        self.gap_detector = gap_detector or GapDetector(db_session)
        self.competitor_analyzer = competitor_analyzer or CompetitorAnalyzer(db_session)
        self.recommendation_engine = recommendation_engine or ContentRecommendationEngine()
        self.calendar_generator = calendar_generator or CalendarGenerator()
        self.benchmark_tracker = benchmark_tracker or BenchmarkTracker()
        self.opportunity_scorer = opportunity_scorer or OpportunityScorer()

    async def analyze_gaps(
        self,
        user_id: str,
        platform: str,
        competitor_ids: Optional[List[str]] = None,
        industry: Optional[str] = None,
        lookback_days: int = 90,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive gap analysis.

        Args:
            user_id: User identifier
            platform: Platform to analyze
            competitor_ids: Optional competitors
            industry: Optional industry
            lookback_days: Days of history
            include_recommendations: Generate recommendations

        Returns:
            Complete gap analysis report

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        logger.info(
            f"Starting gap analysis for user {user_id} on {platform}"
        )

        # Extract topics
        topics = await self.topic_extractor.extract(
            user_id=user_id,
            platform=platform,
            lookback_days=lookback_days
        )

        # Analyze coverage
        coverage = await self.coverage_mapper.analyze(
            user_id=user_id,
            platform=platform,
            lookback_days=lookback_days
        )

        # Detect gaps
        gap_analysis = await self.gap_detector.detect(
            user_id=user_id,
            platform=platform,
            competitor_ids=competitor_ids,
            industry=industry,
            lookback_days=lookback_days
        )

        # Competitor analysis (if competitors provided)
        competitive_analysis = None
        if competitor_ids:
            competitive_analysis = await self.competitor_analyzer.analyze(
                user_id=user_id,
                competitor_ids=competitor_ids,
                platform=platform,
                lookback_days=lookback_days
            )

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            all_gaps = (
                gap_analysis.critical_gaps +
                gap_analysis.important_gaps +
                gap_analysis.moderate_gaps
            )

            recommendations = self.recommendation_engine.generate(
                gaps=all_gaps,
                count=20
            )

        # Build comprehensive report
        report = {
            "user_id": user_id,
            "platform": platform,
            "analysis_date": datetime.now().isoformat(),
            "lookback_days": lookback_days,

            # Topics
            "topics_found": len(topics),
            "topics": [t.to_dict() for t in topics[:10]],

            # Coverage
            "coverage_score": coverage.coverage_percentage,
            "coverage_details": coverage.to_dict(),

            # Gaps
            "total_gaps": gap_analysis.total_gaps,
            "critical_gaps": [g.to_dict() for g in gap_analysis.critical_gaps],
            "important_gaps": [g.to_dict() for g in gap_analysis.important_gaps],
            "moderate_gaps": [g.to_dict() for g in gap_analysis.moderate_gaps],
            "quick_wins": [g.to_dict() for g in gap_analysis.quick_wins],
            "strategic_opportunities": [g.to_dict() for g in gap_analysis.strategic_opportunities],
            "immediate_actions": gap_analysis.immediate_actions,

            # Competitive
            "competitive_position": gap_analysis.competitive_position,
            "competitive_analysis": competitive_analysis.to_dict() if competitive_analysis else None,

            # Recommendations
            "recommendations": [r.to_dict() for r in recommendations],
            "recommendations_count": len(recommendations),
        }

        logger.info(
            f"Gap analysis complete: {gap_analysis.total_gaps} gaps, "
            f"{len(recommendations)} recommendations"
        )

        return report

    async def generate_calendar(
        self,
        user_id: str,
        platform: str,
        weeks: int = 4,
        posts_per_week: int = 3,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate optimized content calendar.

        Args:
            user_id: User identifier
            platform: Target platform
            weeks: Number of weeks
            posts_per_week: Target posting frequency
            start_date: Calendar start date

        Returns:
            Content calendar

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        logger.info(
            f"Generating {weeks}-week calendar for user {user_id} on {platform}"
        )

        # Get gap analysis first
        gap_report = await self.analyze_gaps(
            user_id=user_id,
            platform=platform,
            include_recommendations=True
        )

        # Extract recommendations
        from bufferiq.ml.gaps.recommendations.generator import ContentRecommendation

        recommendations = []
        for rec_dict in gap_report["recommendations"]:
            # Reconstruct recommendation objects
            rec = ContentRecommendation(
                recommendation_id=rec_dict["recommendation_id"],
                topic=rec_dict["topic"],
                title_suggestions=rec_dict["title_suggestions"],
                recommended_format=rec_dict["recommended_format"],
                suggested_length=rec_dict["suggested_length"],
                key_points=rec_dict["key_points"],
                suggested_angles=rec_dict["suggested_angles"],
                optimal_platform=rec_dict["optimal_platform"],
                priority_score=rec_dict["priority_score"],
                estimated_engagement=rec_dict["estimated_engagement"],
            )
            recommendations.append(rec)

        # Generate calendar
        calendar = self.calendar_generator.generate(
            recommendations=recommendations,
            weeks=weeks,
            posts_per_week=posts_per_week,
            start_date=start_date,
            platform=platform
        )

        logger.info(
            f"Calendar generated: {calendar.total_pieces} pieces"
        )

        return calendar.to_dict()

    async def benchmark_competitors(
        self,
        user_id: str,
        competitor_ids: List[str],
        platform: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark against competitors.

        Args:
            user_id: User identifier
            competitor_ids: Competitors to analyze
            platform: Platform
            metrics: Optional specific metrics

        Returns:
            Competitive analysis

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        logger.info(
            f"Benchmarking user {user_id} against {len(competitor_ids)} competitors"
        )

        # Run competitive analysis
        analysis = await self.competitor_analyzer.analyze(
            user_id=user_id,
            competitor_ids=competitor_ids,
            platform=platform
        )

        # Track benchmarks
        user_metrics = {
            "engagement_rate": analysis.user_profile.avg_engagement_rate,
            "posts_per_week": analysis.user_profile.posts_per_week,
            "total_posts": float(analysis.user_profile.total_posts),
        }

        benchmark_tracking = self.benchmark_tracker.track(
            user_id=user_id,
            metrics=user_metrics,
            timestamp=datetime.now()
        )

        return {
            "competitive_analysis": analysis.to_dict(),
            "benchmark_tracking": benchmark_tracking,
        }

    async def get_quick_insights(
        self,
        user_id: str,
        platform: str,
    ) -> Dict[str, Any]:
        """
        Get quick gap insights (cached, fast).

        Args:
            user_id: User identifier
            platform: Platform

        Returns:
            Quick insights summary
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Check cache
        cache_key = f"gap_insights:{user_id}:{platform}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Returning cached insights")
                return cached

        # Run quick analysis
        gap_analysis = await self.gap_detector.detect(
            user_id=user_id,
            platform=platform,
            lookback_days=30  # Shorter for quick insights
        )

        insights = {
            "total_gaps": gap_analysis.total_gaps,
            "critical_count": len(gap_analysis.critical_gaps),
            "coverage_score": gap_analysis.coverage_score,
            "competitive_position": gap_analysis.competitive_position,
            "top_opportunities": [
                g.to_dict() for g in gap_analysis.quick_wins[:3]
            ],
        }

        # Cache for 1 hour
        if self.cache:
            self.cache.setex(cache_key, 3600, insights)

        return insights