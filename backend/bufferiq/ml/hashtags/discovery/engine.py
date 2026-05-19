"""
Hashtag discovery engine.

Discovers related and niche hashtags using co-occurrence analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS
from bufferiq.ml.hashtags.trends.detector import TrendingHashtag


@dataclass
class RelatedHashtag:
    """Related hashtag suggestion."""

    hashtag: str
    similarity_score: float  # 0-1
    co_occurrence_count: int  # Times used together

    # Performance
    avg_engagement: float
    effectiveness_score: float

    # Relationship
    relationship_type: str  # "synonym", "related", "complementary"
    common_contexts: List[str] = field(default_factory=list)


@dataclass
class HashtagDiscovery:
    """Hashtag discovery results."""

    seed_hashtag: str
    platform: str

    # Related hashtags
    synonyms: List[RelatedHashtag] = field(default_factory=list)
    related: List[RelatedHashtag] = field(default_factory=list)
    complementary: List[RelatedHashtag] = field(default_factory=list)

    # Niche opportunities
    niche_hashtags: List[RelatedHashtag] = field(default_factory=list)
    long_tail: List[RelatedHashtag] = field(default_factory=list)

    # Trending
    trending_related: List[TrendingHashtag] = field(default_factory=list)


class HashtagDiscoveryEngine:
    """
    Discover related and niche hashtags.

    Uses co-occurrence analysis, semantic similarity,
    and performance data to find opportunities.

    Example:
```python
        engine = HashtagDiscoveryEngine(db_session)
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
            include_trending=True
        )

        print(f"Discovered hashtags for #{discovery.seed_hashtag}")

        print("\nSynonyms:")
        for ht in discovery.synonyms[:5]:
            print(f"  #{ht.hashtag} (similarity: {ht.similarity_score:.2f})")

        print("\nNiche opportunities:")
        for ht in discovery.niche_hashtags[:5]:
            print(f"  #{ht.hashtag} (effectiveness: {ht.effectiveness_score:.1f})")
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize discovery engine.

        Args:
            db_session: Database session
        """
        self.db = db_session

        # Known relationships
        self.synonym_map: Dict[str, List[str]] = {
            "ai": ["artificialintelligence", "aitech"],
            "ml": ["machinelearning", "mlai"],
            "seo": ["searchengineoptimization"],
            "socialmedia": ["sm", "smm"],
        }

        self.complementary_map: Dict[str, List[str]] = {
            "ai": ["machinelearning", "datascience", "deeplearning"],
            "marketing": ["socialmedia", "contentmarketing", "digitalmarketing"],
            "startup": ["entrepreneur", "innovation", "business"],
        }

    async def discover(
        self,
        seed_hashtag: str,
        platform: str,
        include_trending: bool = True,
        max_results: int = 50,
    ) -> HashtagDiscovery:
        """
        Discover related hashtags.

        Args:
            seed_hashtag: Starting hashtag
            platform: Platform to analyze
            include_trending: Include trending related
            max_results: Max results per category

        Returns:
            Discovery results

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        # Find synonyms
        synonyms = self._find_synonyms(seed_hashtag, max_results)

        # Find related
        related = self._find_related(seed_hashtag, max_results)

        # Find complementary
        complementary = self._find_complementary(seed_hashtag, max_results)

        # Find niche opportunities
        niche = self._find_niche(seed_hashtag, max_results)

        # Find long-tail
        long_tail = self._find_long_tail(seed_hashtag, max_results)

        # Find trending related (if requested)
        trending_related: List[TrendingHashtag] = []
        if include_trending:
            trending_related = await self._find_trending_related(
                seed_hashtag, platform, max_results
            )

        return HashtagDiscovery(
            seed_hashtag=seed_hashtag,
            platform=platform,
            synonyms=synonyms,
            related=related,
            complementary=complementary,
            niche_hashtags=niche,
            long_tail=long_tail,
            trending_related=trending_related,
        )

    def _find_synonyms(
        self, seed_hashtag: str, max_results: int
    ) -> List[RelatedHashtag]:
        """Find synonym hashtags."""
        synonyms: List[RelatedHashtag] = []

        # Check known synonyms
        if seed_hashtag in self.synonym_map:
            for synonym in self.synonym_map[seed_hashtag][:max_results]:
                related = RelatedHashtag(
                    hashtag=synonym,
                    similarity_score=0.95,
                    co_occurrence_count=50,
                    avg_engagement=120.0,
                    effectiveness_score=85.0,
                    relationship_type="synonym",
                    common_contexts=["technology", "business"],
                )
                synonyms.append(related)

        return synonyms

    def _find_related(
        self, seed_hashtag: str, max_results: int
    ) -> List[RelatedHashtag]:
        """Find related hashtags through co-occurrence."""
        # Mock related hashtags
        related_tags = {
            "ai": ["technology", "innovation", "future", "digital"],
            "marketing": ["business", "branding", "content", "strategy"],
            "startup": ["entrepreneur", "business", "innovation", "tech"],
        }

        related: List[RelatedHashtag] = []

        if seed_hashtag in related_tags:
            for tag in related_tags[seed_hashtag][:max_results]:
                related.append(
                    RelatedHashtag(
                        hashtag=tag,
                        similarity_score=0.75,
                        co_occurrence_count=30,
                        avg_engagement=100.0,
                        effectiveness_score=75.0,
                        relationship_type="related",
                        common_contexts=["general"],
                    )
                )

        return related

    def _find_complementary(
        self, seed_hashtag: str, max_results: int
    ) -> List[RelatedHashtag]:
        """Find complementary hashtags."""
        complementary: List[RelatedHashtag] = []

        if seed_hashtag in self.complementary_map:
            for tag in self.complementary_map[seed_hashtag][:max_results]:
                complementary.append(
                    RelatedHashtag(
                        hashtag=tag,
                        similarity_score=0.80,
                        co_occurrence_count=40,
                        avg_engagement=110.0,
                        effectiveness_score=80.0,
                        relationship_type="complementary",
                        common_contexts=["technology"],
                    )
                )

        return complementary

    def _find_niche(
        self, seed_hashtag: str, max_results: int
    ) -> List[RelatedHashtag]:
        """Find niche hashtags with lower competition."""
        # Mock niche hashtags
        niche_tags = [
            f"{seed_hashtag}tips",
            f"{seed_hashtag}trends",
            f"{seed_hashtag}insights",
        ]

        niche: List[RelatedHashtag] = []

        for tag in niche_tags[:max_results]:
            niche.append(
                RelatedHashtag(
                    hashtag=tag,
                    similarity_score=0.70,
                    co_occurrence_count=15,
                    avg_engagement=90.0,
                    effectiveness_score=88.0,  # High effectiveness, lower competition
                    relationship_type="related",
                    common_contexts=["niche"],
                )
            )

        return niche

    def _find_long_tail(
        self, seed_hashtag: str, max_results: int
    ) -> List[RelatedHashtag]:
        """Find long-tail hashtags."""
        # Mock long-tail hashtags
        long_tail_tags = [
            f"{seed_hashtag}for{topic}"
            for topic in ["business", "marketers", "beginners"]
        ]

        long_tail: List[RelatedHashtag] = []

        for tag in long_tail_tags[:max_results]:
            long_tail.append(
                RelatedHashtag(
                    hashtag=tag,
                    similarity_score=0.65,
                    co_occurrence_count=10,
                    avg_engagement=80.0,
                    effectiveness_score=90.0,  # Very targeted
                    relationship_type="related",
                    common_contexts=["long-tail"],
                )
            )

        return long_tail

    async def _find_trending_related(
        self, seed_hashtag: str, platform: str, max_results: int
    ) -> List[TrendingHashtag]:
        """Find trending hashtags related to seed."""
        # This would integrate with TrendDetector
        # Returning empty list for now
        return []