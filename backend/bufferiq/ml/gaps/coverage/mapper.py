"""Content coverage mapping and analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.gaps.coverage.saturation_analyzer import SaturationAnalyzer
from bufferiq.ml.gaps.coverage.diversity_scorer import DiversityScorer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class CoverageMap:
    """Content coverage analysis."""

    total_topics: int
    covered_topics: int
    coverage_percentage: float

    # By category
    topic_distribution: Dict[str, int] = field(default_factory=dict)
    saturation_scores: Dict[str, float] = field(default_factory=dict)

    # Publishing patterns
    posts_per_topic: Dict[str, int] = field(default_factory=dict)
    avg_posts_per_topic: float = 0.0
    most_covered: List[Tuple[str, int]] = field(default_factory=list)
    least_covered: List[Tuple[str, int]] = field(default_factory=list)

    # Platform distribution
    platform_coverage: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Content type diversity
    content_type_diversity: float = 0.0
    format_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_topics": self.total_topics,
            "covered_topics": self.covered_topics,
            "coverage_percentage": self.coverage_percentage,
            "topic_distribution": self.topic_distribution,
            "saturation_scores": self.saturation_scores,
            "posts_per_topic": self.posts_per_topic,
            "avg_posts_per_topic": self.avg_posts_per_topic,
            "most_covered": self.most_covered,
            "least_covered": self.least_covered,
            "platform_coverage": self.platform_coverage,
            "content_type_diversity": self.content_type_diversity,
            "format_distribution": self.format_distribution,
        }


class CoverageMapper:
    """
    Map content coverage across topics and platforms.

    Analyzes what topics are covered, how frequently,
    and identifies coverage imbalances.

    Example:
```python
        mapper = CoverageMapper(db_session)
        coverage = await mapper.analyze(
            user_id="user123",
            platform="linkedin",
            lookback_days=90
        )

        print(f"Coverage: {coverage.coverage_percentage:.1f}%")
        print(f"Topics covered: {coverage.covered_topics}/{coverage.total_topics}")

        for topic, count in coverage.least_covered:
            print(f"  {topic}: only {count} posts")
```
    """

    def __init__(self, db_session: Session):
        """
        Initialize coverage mapper.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self.saturation_analyzer = SaturationAnalyzer()
        self.diversity_scorer = DiversityScorer()

    async def analyze(
        self,
        user_id: str,
        platform: str,
        lookback_days: int = 90,
        industry_topics: Optional[List[str]] = None,
    ) -> CoverageMap:
        """
        Analyze content coverage.

        Args:
            user_id: User identifier
            platform: Platform to analyze
            lookback_days: Days of history
            industry_topics: Optional industry topic list

        Returns:
            Coverage analysis

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        logger.info(f"Analyzing coverage for user {user_id} on {platform}")

        # Fetch user's posts and topics
        user_topics = await self._fetch_user_topics(user_id, platform, lookback_days)

        # Get industry benchmark topics if not provided
        if industry_topics is None:
            industry_topics = self._get_industry_topics(platform)

        # Calculate coverage metrics
        covered_topics = len(user_topics)
        total_topics = len(industry_topics)
        coverage_percentage = (
            (covered_topics / total_topics * 100) if total_topics > 0 else 0.0
        )

        # Topic distribution
        topic_distribution = self._calculate_topic_distribution(user_topics)

        # Saturation scores
        saturation_scores = self.saturation_analyzer.calculate_saturation(user_topics)

        # Posts per topic
        posts_per_topic = {topic["name"]: topic["post_count"] for topic in user_topics}
        avg_posts = (
            sum(posts_per_topic.values()) / len(posts_per_topic)
            if posts_per_topic
            else 0.0
        )

        # Most/least covered
        sorted_topics = sorted(posts_per_topic.items(), key=lambda x: x[1], reverse=True)
        most_covered = sorted_topics[:5]
        least_covered = sorted_topics[-5:][::-1]

        # Platform coverage
        platform_coverage = await self._calculate_platform_coverage(
            user_id, lookback_days
        )

        # Content diversity
        content_types = [topic.get("content_type", "article") for topic in user_topics]
        content_type_diversity = self.diversity_scorer.calculate_diversity(content_types)
        format_distribution = self._calculate_format_distribution(user_topics)

        return CoverageMap(
            total_topics=total_topics,
            covered_topics=covered_topics,
            coverage_percentage=round(coverage_percentage, 2),
            topic_distribution=topic_distribution,
            saturation_scores=saturation_scores,
            posts_per_topic=posts_per_topic,
            avg_posts_per_topic=round(avg_posts, 2),
            most_covered=most_covered,
            least_covered=least_covered,
            platform_coverage=platform_coverage,
            content_type_diversity=round(content_type_diversity, 3),
            format_distribution=format_distribution,
        )

    async def _fetch_user_topics(
        self, user_id: str, platform: str, lookback_days: int
    ) -> List[Dict[str, Any]]:
        """Fetch user's topics (mock implementation)."""
        # Mock data
        return [
            {
                "name": "AI & Machine Learning",
                "post_count": 15,
                "category": "technology",
                "content_type": "article",
            },
            {
                "name": "Python Programming",
                "post_count": 12,
                "category": "technology",
                "content_type": "tutorial",
            },
            {
                "name": "Data Science",
                "post_count": 10,
                "category": "analytics",
                "content_type": "article",
            },
            {
                "name": "Leadership",
                "post_count": 8,
                "category": "management",
                "content_type": "opinion",
            },
            {
                "name": "Innovation",
                "post_count": 7,
                "category": "business",
                "content_type": "article",
            },
        ]

    def _get_industry_topics(self, platform: str) -> List[str]:
        """Get industry standard topics."""
        # Industry benchmark topics
        return [
            "AI & Machine Learning",
            "Python Programming",
            "Data Science",
            "Leadership",
            "Innovation",
            "Cloud Computing",
            "Cybersecurity",
            "Digital Transformation",
            "Product Management",
            "Team Building",
            "Remote Work",
            "Career Development",
        ]

    def _calculate_topic_distribution(
        self, topics: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate topic distribution by category."""
        distribution: Dict[str, int] = {}
        for topic in topics:
            category = topic.get("category", "other")
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    async def _calculate_platform_coverage(
        self, user_id: str, lookback_days: int
    ) -> Dict[str, Dict[str, float]]:
        """Calculate coverage across platforms."""
        # Mock data
        return {
            "linkedin": {"coverage": 0.75, "posts": 35},
            "twitter": {"coverage": 0.45, "posts": 12},
            "bluesky": {"coverage": 0.30, "posts": 8},
        }

    def _calculate_format_distribution(
        self, topics: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate content format distribution."""
        distribution: Dict[str, int] = {}
        for topic in topics:
            fmt = topic.get("content_type", "article")
            distribution[fmt] = distribution.get(fmt, 0) + 1
        return distribution