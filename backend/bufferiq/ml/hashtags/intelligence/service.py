"""
Hashtag Intelligence Service.

Main orchestrator for all hashtag analysis functionality.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from bufferiq.ml.hashtags.extraction.extractor import (
    HashtagExtractor,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.hashtags.performance.analyzer import (
    HashtagPerformanceAnalyzer,
    HashtagPerformance,
)
from bufferiq.ml.hashtags.trends.detector import (
    TrendDetector,
    TrendingHashtag,
)
from bufferiq.ml.hashtags.discovery.engine import (
    HashtagDiscoveryEngine,
    HashtagDiscovery,
)
from bufferiq.ml.hashtags.strategy.generator import (
    HashtagStrategyGenerator,
    HashtagStrategy,
)
from bufferiq.ml.hashtags.risks.detector import (
    HashtagRiskDetector,
    HashtagRisk,
)


class HashtagIntelligenceService:
    """
    Main orchestrator for hashtag intelligence.

    Coordinates all hashtag analysis modules:
    - Extraction
    - Performance analysis
    - Trend detection
    - Discovery
    - Strategy generation
    - Effectiveness scoring
    - Risk detection
    - Combination optimization

    Example:
```python
        service = HashtagIntelligenceService(
            db_session=session,
            cache=redis_client
        )

        # Analyze hashtag performance
        performance = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
            user_id="user123"
        )

        # Get recommendations
        recommendations = await service.recommend_hashtags(
            content="Great insights on AI and ML",
            platform="linkedin",
            user_id="user123",
            count=5
        )

        # Get trending
        trending = await service.get_trending(
            platform="linkedin",
            category="technology"
        )

        # Validate safety
        validation = await service.validate_hashtags(
            hashtags=["ai", "tech", "innovation"],
            platform="linkedin"
        )
```
    """

    def __init__(
        self,
        db_session: Session,
        cache: Optional[Any] = None,
        extractor: Optional[HashtagExtractor] = None,
        performance_analyzer: Optional[HashtagPerformanceAnalyzer] = None,
        trend_detector: Optional[TrendDetector] = None,
        discovery_engine: Optional[HashtagDiscoveryEngine] = None,
        strategy_generator: Optional[HashtagStrategyGenerator] = None,
        risk_detector: Optional[HashtagRiskDetector] = None,
    ) -> None:
        """
        Initialize hashtag intelligence service.

        Args:
            db_session: Database session
            cache: Optional cache (e.g., Redis)
            extractor: Optional custom extractor
            performance_analyzer: Optional custom analyzer
            trend_detector: Optional custom detector
            discovery_engine: Optional custom engine
            strategy_generator: Optional custom generator
            risk_detector: Optional custom detector
        """
        self.db = db_session
        self.cache = cache

        # Initialize components
        self.extractor = extractor or HashtagExtractor()
        self.performance_analyzer = performance_analyzer or HashtagPerformanceAnalyzer(
            db_session
        )
        self.trend_detector = trend_detector or TrendDetector(db_session)
        self.discovery_engine = discovery_engine or HashtagDiscoveryEngine(db_session)
        self.strategy_generator = strategy_generator or HashtagStrategyGenerator()
        self.risk_detector = risk_detector or HashtagRiskDetector()

    async def analyze_hashtag(
        self,
        hashtag: str,
        platform: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive hashtag analysis.

        Args:
            hashtag: Hashtag to analyze
            platform: Platform name
            user_id: Optional user context

        Returns:
            Complete analysis results

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Performance analysis
        performance = await self.performance_analyzer.analyze(
            hashtag=hashtag,
            platform=platform,
            user_id=user_id,
        )

        # Risk assessment
        risk = self.risk_detector.assess(
            hashtag=hashtag,
            platform=platform,
        )

        # Discovery (related hashtags)
        discovery = await self.discovery_engine.discover(
            seed_hashtag=hashtag,
            platform=platform,
            include_trending=False,
        )

        return {
            "hashtag": hashtag,
            "platform": platform,
            "performance": {
                "total_uses": performance.total_uses,
                "avg_engagement": performance.avg_engagement,
                "engagement_lift": performance.engagement_lift,
                "trend_direction": performance.trend_direction,
                "roi": performance.estimated_roi,
            },
            "risk": {
                "risk_level": risk.risk_level,
                "is_safe": risk.risk_level in ["none", "low"],
                "reasons": risk.risk_reasons,
                "recommendation": risk.recommendation,
            },
            "related": {
                "synonyms": [
                    {"hashtag": h.hashtag, "score": h.similarity_score}
                    for h in discovery.synonyms[:5]
                ],
                "complementary": [
                    {"hashtag": h.hashtag, "score": h.similarity_score}
                    for h in discovery.complementary[:5]
                ],
            },
        }

    async def recommend_hashtags(
        self,
        content: str,
        platform: str,
        user_id: Optional[str] = None,
        count: int = 5,
    ) -> List[str]:
        """
        Recommend hashtags for content.

        Args:
            content: Content text
            platform: Target platform
            user_id: Optional user context
            count: Number of recommendations

        Returns:
            List of recommended hashtags
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Generate strategy
        strategy = self.strategy_generator.generate(
            platform=platform,
            content_topic=content,
        )

        # Return recommended hashtags
        return strategy.recommended_hashtags[:count]

    async def get_trending(
        self,
        platform: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[TrendingHashtag]:
        """
        Get trending hashtags.

        Args:
            platform: Platform to check
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of trending hashtags
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        trending = await self.trend_detector.detect_trending(
            platform=platform,
            category=category,
            limit=limit,
        )

        return trending

    async def validate_hashtags(
        self,
        hashtags: List[str],
        platform: str,
    ) -> Dict[str, HashtagRisk]:
        """
        Validate hashtag safety.

        Args:
            hashtags: List of hashtags to validate
            platform: Platform name

        Returns:
            Map of hashtag -> risk assessment
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        validation: Dict[str, HashtagRisk] = {}

        for hashtag in hashtags:
            risk = self.risk_detector.assess(
                hashtag=hashtag,
                platform=platform,
            )
            validation[hashtag] = risk

        return validation

    async def generate_strategy(
        self,
        content: str,
        platform: str,
        user_profile: Optional[Any] = None,
    ) -> HashtagStrategy:
        """
        Generate complete hashtag strategy.

        Args:
            content: Content text
            platform: Target platform
            user_profile: Optional user profile

        Returns:
            Complete strategy
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        strategy = self.strategy_generator.generate(
            platform=platform,
            content_topic=content,
            user_profile=user_profile,
        )

        return strategy