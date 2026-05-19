"""Tests for discovery engine."""

import pytest
from unittest.mock import Mock, AsyncMock

from bufferiq.ml.hashtags.discovery.engine import (
    HashtagDiscoveryEngine,
    HashtagDiscovery,
    RelatedHashtag,
)


class TestHashtagDiscoveryEngine:
    """Test HashtagDiscoveryEngine class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def engine(self, mock_db):
        """Create engine instance."""
        return HashtagDiscoveryEngine(db_session=mock_db)

    @pytest.mark.asyncio
    async def test_discover_basic(self, engine):
        """Test basic discovery."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(discovery, HashtagDiscovery)
        assert discovery.seed_hashtag == "ai"
        assert discovery.platform == "linkedin"

    @pytest.mark.asyncio
    async def test_discover_synonyms(self, engine):
        """Test synonym discovery."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(discovery.synonyms, list)
        assert all(isinstance(h, RelatedHashtag) for h in discovery.synonyms)

    @pytest.mark.asyncio
    async def test_discover_related(self, engine):
        """Test related hashtag discovery."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(discovery.related, list)

    @pytest.mark.asyncio
    async def test_discover_complementary(self, engine):
        """Test complementary discovery."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(discovery.complementary, list)

    @pytest.mark.asyncio
    async def test_discover_niche(self, engine):
        """Test niche discovery."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(discovery.niche_hashtags, list)

    @pytest.mark.asyncio
    async def test_discover_invalid_platform(self, engine):
        """Test with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await engine.discover(
                seed_hashtag="ai",
                platform="facebook",
            )

    @pytest.mark.asyncio
    async def test_discover_max_results(self, engine):
        """Test max results limit."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
            max_results=5,
        )

        # Each category should respect max_results
        assert len(discovery.synonyms) <= 5
        assert len(discovery.related) <= 5

    @pytest.mark.asyncio
    async def test_related_hashtag_structure(self, engine):
        """Test RelatedHashtag structure."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        if discovery.synonyms:
            related = discovery.synonyms[0]
            assert hasattr(related, "hashtag")
            assert hasattr(related, "similarity_score")
            assert hasattr(related, "co_occurrence_count")
            assert hasattr(related, "effectiveness_score")
            assert hasattr(related, "relationship_type")

    @pytest.mark.asyncio
    async def test_similarity_scores(self, engine):
        """Test similarity scores are valid."""
        discovery = await engine.discover(
            seed_hashtag="ai",
            platform="linkedin",
        )

        for hashtag in discovery.synonyms:
            assert 0 <= hashtag.similarity_score <= 1


class TestRelatedHashtagFinder:
    """Test RelatedHashtagFinder class."""

    @pytest.fixture
    def finder(self):
        """Create finder instance."""
        from bufferiq.ml.hashtags.discovery.related_finder import RelatedHashtagFinder
        return RelatedHashtagFinder()

    def test_add_post(self, finder):
        """Test adding post."""
        finder.add_post(["ai", "tech", "innovation"])

        assert finder.hashtag_counts["ai"] == 1
        assert finder.hashtag_counts["tech"] == 1

    def test_find_related(self, finder):
        """Test finding related hashtags."""
        # Add posts with co-occurring hashtags
        finder.add_post(["ai", "tech"])
        finder.add_post(["ai", "innovation"])
        finder.add_post(["ai", "tech"])

        related = finder.find_related("ai", min_score=0.0)

        assert isinstance(related, list)
        assert len(related) > 0

    def test_find_related_empty(self, finder):
        """Test with no data."""
        related = finder.find_related("ai")

        assert related == []

    def test_calculate_similarity(self, finder):
        """Test similarity calculation."""
        finder.add_post(["ai", "tech"])
        finder.add_post(["ai", "tech"])
        finder.add_post(["tech"])

        similarity = finder._calculate_similarity("ai", "tech")

        assert 0 <= similarity <= 1