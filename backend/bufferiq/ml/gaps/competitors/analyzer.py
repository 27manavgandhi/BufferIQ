"""Competitor content analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.gaps.competitors.benchmarker import CompetitorBenchmarker
from bufferiq.ml.gaps.competitors.strategy_detector import StrategyDetector
from bufferiq.ml.gaps.competitors.overlap_analyzer import OverlapAnalyzer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class CompetitorProfile:
    """Competitor content profile."""

    competitor_id: str
    name: str
    platform: str

    # Publishing metrics
    total_posts: int
    posts_per_week: float
    avg_engagement_rate: float

    # Content strategy
    top_topics: List[Tuple[str, int]] = field(default_factory=list)
    content_types: Dict[str, int] = field(default_factory=dict)
    posting_schedule: Dict[str, int] = field(default_factory=dict)

    # Performance
    best_performing_topics: List[Tuple[str, float]] = field(default_factory=list)
    engagement_trend: str = "stable"

    # Voice characteristics
    avg_formality: float = 0.5
    sentiment_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competitor_id": self.competitor_id,
            "name": self.name,
            "platform": self.platform,
            "total_posts": self.total_posts,
            "posts_per_week": self.posts_per_week,
            "avg_engagement_rate": self.avg_engagement_rate,
            "top_topics": self.top_topics,
            "content_types": self.content_types,
            "posting_schedule": self.posting_schedule,
            "best_performing_topics": self.best_performing_topics,
            "engagement_trend": self.engagement_trend,
            "avg_formality": self.avg_formality,
            "sentiment_distribution": self.sentiment_distribution,
        }


@dataclass
class CompetitiveAnalysis:
    """Competitive intelligence analysis."""

    user_profile: CompetitorProfile
    competitor_profiles: List[CompetitorProfile] = field(default_factory=list)

    # Comparative metrics
    user_rank: int = 1
    share_of_voice: float = 0.0
    engagement_vs_avg: float = 1.0

    # Strategic insights
    unique_topics: List[str] = field(default_factory=list)
    missed_topics: List[str] = field(default_factory=list)
    common_topics: List[str] = field(default_factory=list)

    # Opportunities
    competitor_gaps: List[str] = field(default_factory=list)
    differentiation_opportunities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_profile": self.user_profile.to_dict(),
            "competitor_profiles": [c.to_dict() for c in self.competitor_profiles],
            "user_rank": self.user_rank,
            "share_of_voice": self.share_of_voice,
            "engagement_vs_avg": self.engagement_vs_avg,
            "unique_topics": self.unique_topics,
            "missed_topics": self.missed_topics,
            "common_topics": self.common_topics,
            "competitor_gaps": self.competitor_gaps,
            "differentiation_opportunities": self.differentiation_opportunities,
        }


class CompetitorAnalyzer:
    """
    Analyze competitor content strategies.

    Benchmarks user against competitors to identify
    strategic opportunities and gaps.

    Example:
```python
        analyzer = CompetitorAnalyzer(db_session)
        analysis = await analyzer.analyze(
            user_id="user123",
            competitor_ids=["comp1", "comp2", "comp3"],
            platform="linkedin"
        )

        print(f"Your rank: {analysis.user_rank}/{len(analysis.competitor_profiles)+1}")
        print(f"Share of voice: {analysis.share_of_voice:.1f}%")

        print(f"\\nYour unique topics:")
        for topic in analysis.unique_topics:
            print(f"  - {topic}")

        print(f"\\nMissed opportunities:")
        for topic in analysis.missed_topics:
            print(f"  - {topic}")
```
    """

    def __init__(self, db_session: Session):
        """
        Initialize competitor analyzer.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self.benchmarker = CompetitorBenchmarker()
        self.strategy_detector = StrategyDetector()
        self.overlap_analyzer = OverlapAnalyzer()

    async def analyze(
        self,
        user_id: str,
        competitor_ids: List[str],
        platform: str,
        lookback_days: int = 90,
    ) -> CompetitiveAnalysis:
        """
        Analyze competitive landscape.

        Args:
            user_id: User identifier
            competitor_ids: List of competitor IDs
            platform: Platform to analyze
            lookback_days: Days of history

        Returns:
            Competitive analysis

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        logger.info(
            f"Analyzing {len(competitor_ids)} competitors for user {user_id} on {platform}"
        )

        # Build user profile
        user_profile = await self._build_profile(user_id, platform, lookback_days)

        # Build competitor profiles
        competitor_profiles = []
        for comp_id in competitor_ids:
            profile = await self._build_profile(comp_id, platform, lookback_days)
            competitor_profiles.append(profile)

        # Benchmark user vs competitors
        benchmark_results = self.benchmarker.benchmark(user_profile, competitor_profiles)

        # Detect strategy patterns
        strategies = self.strategy_detector.detect(competitor_profiles)

        # Analyze topic overlap
        overlap = self.overlap_analyzer.analyze(user_profile, competitor_profiles)

        # Calculate share of voice
        total_posts = user_profile.total_posts + sum(
            c.total_posts for c in competitor_profiles
        )
        share_of_voice = (
            (user_profile.total_posts / total_posts * 100) if total_posts > 0 else 0
        )

        # Rank user
        all_profiles = [user_profile] + competitor_profiles
        sorted_profiles = sorted(
            all_profiles, key=lambda p: p.avg_engagement_rate, reverse=True
        )
        user_rank = next(
            i + 1
            for i, p in enumerate(sorted_profiles)
            if p.competitor_id == user_id
        )

        # Engagement vs average
        avg_competitor_engagement = (
            sum(c.avg_engagement_rate for c in competitor_profiles)
            / len(competitor_profiles)
            if competitor_profiles
            else 1.0
        )
        engagement_vs_avg = (
            user_profile.avg_engagement_rate / avg_competitor_engagement
            if avg_competitor_engagement > 0
            else 1.0
        )

        return CompetitiveAnalysis(
            user_profile=user_profile,
            competitor_profiles=competitor_profiles,
            user_rank=user_rank,
            share_of_voice=round(share_of_voice, 2),
            engagement_vs_avg=round(engagement_vs_avg, 2),
            unique_topics=overlap["unique_topics"],
            missed_topics=overlap["missed_topics"],
            common_topics=overlap["common_topics"],
            competitor_gaps=overlap.get("competitor_gaps", []),
            differentiation_opportunities=strategies.get("opportunities", []),
        )

    async def _build_profile(
        self, entity_id: str, platform: str, lookback_days: int
    ) -> CompetitorProfile:
        """Build competitor profile."""
        # Mock implementation
        # In production, fetch real data from database

        import random

        return CompetitorProfile(
            competitor_id=entity_id,
            name=f"Entity {entity_id}",
            platform=platform,
            total_posts=random.randint(30, 100),
            posts_per_week=random.uniform(2.0, 5.0),
            avg_engagement_rate=random.uniform(0.02, 0.08),
            top_topics=[
                ("AI & Machine Learning", random.randint(5, 20)),
                ("Cloud Computing", random.randint(3, 15)),
                ("Data Science", random.randint(2, 12)),
            ],
            content_types={"article": 40, "tutorial": 30, "opinion": 20, "news": 10},
            posting_schedule={"morning": 40, "afternoon": 35, "evening": 25},
            best_performing_topics=[
                ("AI & Machine Learning", random.uniform(200, 500)),
                ("Cloud Computing", random.uniform(150, 400)),
            ],
            engagement_trend=random.choice(["growing", "stable", "declining"]),
            avg_formality=random.uniform(0.4, 0.8),
            sentiment_distribution={
                "positive": random.uniform(0.5, 0.7),
                "neutral": random.uniform(0.2, 0.4),
                "negative": random.uniform(0.0, 0.1),
            },
        )