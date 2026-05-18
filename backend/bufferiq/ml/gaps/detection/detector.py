"""Content gap detection."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.gaps.detection.classifier import GapClassifier
from bufferiq.ml.gaps.detection.prioritizer import GapPrioritizer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class GapSeverity(Enum):
    """Gap severity levels."""

    CRITICAL = "critical"  # Major opportunity, high priority
    IMPORTANT = "important"  # Significant gap
    MODERATE = "moderate"  # Notable absence
    MINOR = "minor"  # Nice-to-have


@dataclass
class ContentGap:
    """Identified content gap."""

    gap_id: str
    topic: str
    keywords: List[str]
    description: str

    # Severity & priority
    severity: GapSeverity
    priority_score: float  # 0-100
    opportunity_score: float  # 0-100

    # Context
    competitor_coverage: int  # How many competitors cover this
    search_volume: Optional[int] = None
    trend_direction: str = "stable"  # "rising", "stable", "falling"

    # Recommendations
    recommended_content_types: List[str] = field(default_factory=list)
    suggested_angles: List[str] = field(default_factory=list)
    estimated_engagement: Optional[float] = None

    # Meta
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_id": self.gap_id,
            "topic": self.topic,
            "keywords": self.keywords,
            "description": self.description,
            "severity": self.severity.value,
            "priority_score": self.priority_score,
            "opportunity_score": self.opportunity_score,
            "competitor_coverage": self.competitor_coverage,
            "search_volume": self.search_volume,
            "trend_direction": self.trend_direction,
            "recommended_content_types": self.recommended_content_types,
            "suggested_angles": self.suggested_angles,
            "estimated_engagement": self.estimated_engagement,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class GapAnalysis:
    """Complete gap analysis results."""

    total_gaps: int
    critical_gaps: List[ContentGap] = field(default_factory=list)
    important_gaps: List[ContentGap] = field(default_factory=list)
    moderate_gaps: List[ContentGap] = field(default_factory=list)

    # Summary metrics
    coverage_score: float = 0.0  # 0-100
    competitive_position: str = "average"  # "leader", "average", "behind"
    total_opportunity_value: float = 0.0

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    quick_wins: List[ContentGap] = field(default_factory=list)
    strategic_opportunities: List[ContentGap] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gaps": self.total_gaps,
            "critical_gaps": [g.to_dict() for g in self.critical_gaps],
            "important_gaps": [g.to_dict() for g in self.important_gaps],
            "moderate_gaps": [g.to_dict() for g in self.moderate_gaps],
            "coverage_score": self.coverage_score,
            "competitive_position": self.competitive_position,
            "total_opportunity_value": self.total_opportunity_value,
            "immediate_actions": self.immediate_actions,
            "quick_wins": [g.to_dict() for g in self.quick_wins],
            "strategic_opportunities": [g.to_dict() for g in self.strategic_opportunities],
        }


class GapDetector:
    """
    Detect content gaps and opportunities.

    Identifies missing topics, under-covered areas,
    and competitive gaps in content strategy.

    Example:
```python
        detector = GapDetector(db_session)
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            competitor_ids=["comp1", "comp2"],
            industry="technology"
        )

        print(f"Found {analysis.total_gaps} gaps")
        print(f"Critical: {len(analysis.critical_gaps)}")

        for gap in analysis.critical_gaps:
            print(f"  {gap.topic}: {gap.description}")
            print(f"    Priority: {gap.priority_score:.1f}")
            print(f"    {gap.competitor_coverage} competitors covering")
```
    """

    def __init__(self, db_session: Session):
        """
        Initialize gap detector.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self.classifier = GapClassifier()
        self.prioritizer = GapPrioritizer()

    async def detect(
        self,
        user_id: str,
        platform: str,
        competitor_ids: Optional[List[str]] = None,
        industry: Optional[str] = None,
        lookback_days: int = 90,
    ) -> GapAnalysis:
        """
        Detect content gaps.

        Args:
            user_id: User identifier
            platform: Platform to analyze
            competitor_ids: Optional competitor list
            industry: Optional industry for benchmarking
            lookback_days: Days of history

        Returns:
            Gap analysis results

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        logger.info(f"Detecting gaps for user {user_id} on {platform}")

        # Get user's topics
        user_topics = await self._fetch_user_topics(user_id, platform, lookback_days)
        user_topic_names = {topic["name"] for topic in user_topics}

        # Get industry/competitor topics
        if competitor_ids:
            benchmark_topics = await self._fetch_competitor_topics(
                competitor_ids, platform, lookback_days
            )
        else:
            benchmark_topics = self._get_industry_topics(industry or "technology")

        # Identify gaps
        gaps = []
        for bench_topic in benchmark_topics:
            topic_name = bench_topic["name"]

            # Check if user covers this topic
            if topic_name not in user_topic_names:
                # This is a gap
                gap = self._create_gap(bench_topic, competitor_ids or [])
                gaps.append(gap)

        # Classify gaps by severity
        classified_gaps = self.classifier.classify(gaps)

        # Prioritize gaps
        prioritized_gaps = self.prioritizer.prioritize(gaps)

        # Separate by severity
        critical_gaps = [g for g in prioritized_gaps if g.severity == GapSeverity.CRITICAL]
        important_gaps = [g for g in prioritized_gaps if g.severity == GapSeverity.IMPORTANT]
        moderate_gaps = [g for g in prioritized_gaps if g.severity == GapSeverity.MODERATE]

        # Calculate coverage score
        total_benchmark = len(benchmark_topics)
        covered = len(user_topic_names)
        coverage_score = (covered / total_benchmark * 100) if total_benchmark > 0 else 0

        # Determine competitive position
        competitive_position = self._determine_position(coverage_score)

        # Calculate opportunity value
        total_opportunity = sum(g.opportunity_score for g in gaps)

        # Identify quick wins (high opportunity, low effort)
        quick_wins = [
            g
            for g in prioritized_gaps
            if g.opportunity_score > 70 and g.priority_score > 80
        ][:5]

        # Strategic opportunities
        strategic = [
            g
            for g in prioritized_gaps
            if g.severity in [GapSeverity.CRITICAL, GapSeverity.IMPORTANT]
        ][:10]

        # Generate immediate actions
        immediate_actions = self._generate_immediate_actions(critical_gaps)

        return GapAnalysis(
            total_gaps=len(gaps),
            critical_gaps=critical_gaps,
            important_gaps=important_gaps,
            moderate_gaps=moderate_gaps,
            coverage_score=round(coverage_score, 2),
            competitive_position=competitive_position,
            total_opportunity_value=round(total_opportunity, 2),
            immediate_actions=immediate_actions,
            quick_wins=quick_wins,
            strategic_opportunities=strategic,
        )

    async def _fetch_user_topics(
        self, user_id: str, platform: str, lookback_days: int
    ) -> List[Dict[str, Any]]:
        """Fetch user's topics."""
        # Mock implementation
        return [
            {"name": "AI & Machine Learning", "post_count": 15},
            {"name": "Python Programming", "post_count": 12},
            {"name": "Data Science", "post_count": 10},
        ]

    async def _fetch_competitor_topics(
        self, competitor_ids: List[str], platform: str, lookback_days: int
    ) -> List[Dict[str, Any]]:
        """Fetch competitor topics."""
        # Mock implementation
        return [
            {
                "name": "AI & Machine Learning",
                "coverage_count": 3,
                "avg_engagement": 250,
                "trend": "rising",
            },
            {
                "name": "Cloud Computing",
                "coverage_count": 4,
                "avg_engagement": 300,
                "trend": "rising",
            },
            {
                "name": "Cybersecurity",
                "coverage_count": 3,
                "avg_engagement": 200,
                "trend": "stable",
            },
            {
                "name": "DevOps",
                "coverage_count": 2,
                "avg_engagement": 180,
                "trend": "growing",
            },
        ]

    def _get_industry_topics(self, industry: str) -> List[Dict[str, Any]]:
        """Get industry standard topics."""
        return [
            {
                "name": "Cloud Computing",
                "coverage_count": 4,
                "avg_engagement": 300,
                "trend": "rising",
            },
            {
                "name": "Cybersecurity",
                "coverage_count": 3,
                "avg_engagement": 200,
                "trend": "stable",
            },
            {
                "name": "DevOps",
                "coverage_count": 2,
                "avg_engagement": 180,
                "trend": "growing",
            },
            {
                "name": "Blockchain",
                "coverage_count": 2,
                "avg_engagement": 150,
                "trend": "stable",
            },
        ]

    def _create_gap(
        self, benchmark_topic: Dict[str, Any], competitor_ids: List[str]
    ) -> ContentGap:
        """Create a content gap from benchmark topic."""
        import hashlib

        topic_name = benchmark_topic["name"]
        coverage_count = benchmark_topic.get("coverage_count", 0)
        avg_engagement = benchmark_topic.get("avg_engagement", 0)
        trend = benchmark_topic.get("trend", "stable")

        # Generate gap ID
        gap_id = hashlib.sha256(topic_name.encode()).hexdigest()[:16]

        # Calculate scores
        opportunity_score = min(
            (coverage_count * 20) + (avg_engagement / 10), 100
        )
        priority_score = opportunity_score * 0.8  # Will be refined by prioritizer

        # Determine severity based on opportunity
        if opportunity_score >= 80:
            severity = GapSeverity.CRITICAL
        elif opportunity_score >= 60:
            severity = GapSeverity.IMPORTANT
        elif opportunity_score >= 40:
            severity = GapSeverity.MODERATE
        else:
            severity = GapSeverity.MINOR

        # Extract keywords
        keywords = topic_name.lower().split()

        # Recommendations
        content_types = ["article", "tutorial", "case_study"]
        angles = [f"Beginner's guide to {topic_name}", f"Best practices in {topic_name}"]

        return ContentGap(
            gap_id=gap_id,
            topic=topic_name,
            keywords=keywords,
            description=f"Missing coverage of {topic_name}",
            severity=severity,
            priority_score=round(priority_score, 2),
            opportunity_score=round(opportunity_score, 2),
            competitor_coverage=coverage_count,
            trend_direction=trend,
            recommended_content_types=content_types,
            suggested_angles=angles,
            estimated_engagement=avg_engagement,
        )

    def _determine_position(self, coverage_score: float) -> str:
        """Determine competitive position."""
        if coverage_score >= 80:
            return "leader"
        elif coverage_score >= 50:
            return "average"
        else:
            return "behind"

    def _generate_immediate_actions(self, critical_gaps: List[ContentGap]) -> List[str]:
        """Generate immediate action items."""
        actions = []

        for gap in critical_gaps[:3]:
            action = f"Create content about {gap.topic} - {gap.competitor_coverage} competitors already covering"
            actions.append(action)

        return actions