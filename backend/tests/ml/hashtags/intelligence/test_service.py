"""Tests for hashtag intelligence service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


class TestHashtagIntelligenceService:
    """Test HashtagIntelligenceService class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def service(self, mock_db):
        """Create service instance."""
        return HashtagIntelligenceService(db_session=mock_db)

    @pytest.mark.asyncio
    async def test_analyze_hashtag(self, service):
        """Test hashtag analysis."""
        analysis = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(analysis, dict)
        assert "hashtag" in analysis
        assert "platform" in analysis
        assert "performance" in analysis
        assert "risk" in analysis
        assert "related" in analysis

    @pytest.mark.asyncio
    async def test_analyze_invalid_platform(self, service):
        """Test analysis with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await service.analyze_hashtag(
                hashtag="ai",
                platform="facebook",
            )

    @pytest.mark.asyncio
    async def test_recommend_hashtags(self, service):
        """Test hashtag recommendations."""
        recommendations = await service.recommend_hashtags(
            content="Great insights on AI and machine learning",
            platform="linkedin",
            count=5,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        assert all(isinstance(h, str) for h in recommendations)

    @pytest.mark.asyncio
    async def test_get_trending(self, service):
        """Test getting trending hashtags."""
        trending = await service.get_trending(
            platform="linkedin",
            limit=20,
        )

        assert isinstance(trending, list)
        assert len(trending) <= 20

    @pytest.mark.asyncio
    async def test_validate_hashtags(self, service):
        """Test hashtag validation."""
        validation = await service.validate_hashtags(
            hashtags=["ai", "tech", "spam"],
            platform="linkedin",
        )

        assert isinstance(validation, dict)
        assert "ai" in validation
        assert "tech" in validation
        assert "spam" in validation

    @pytest.mark.asyncio
    async def test_generate_strategy(self, service):
        """Test strategy generation."""
        strategy = await service.generate_strategy(
            content="AI insights",
            platform="linkedin",
        )

        assert hasattr(strategy, "platform")
        assert hasattr(strategy, "recommended_hashtags")
        assert hasattr(strategy, "recommended_count")

    @pytest.mark.asyncio
    async def test_analyze_with_user(self, service):
        """Test analysis with user context."""
        analysis = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
            user_id="user123",
        )

        assert "performance" in analysis

    @pytest.mark.asyncio
    async def test_performance_structure(self, service):
        """Test performance data structure."""
        analysis = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
        )

        perf = analysis["performance"]
        assert "total_uses" in perf
        assert "avg_engagement" in perf
        assert "engagement_lift" in perf
        assert "trend_direction" in perf
        assert "roi" in perf

    @pytest.mark.asyncio
    async def test_risk_structure(self, service):
        """Test risk data structure."""
        analysis = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
        )

        risk = analysis["risk"]
        assert "risk_level" in risk
        assert "is_safe" in risk
        assert "reasons" in risk
        assert "recommendation" in risk

    @pytest.mark.asyncio
    async def test_related_structure(self, service):
        """Test related data structure."""
        analysis = await service.analyze_hashtag(
            hashtag="ai",
            platform="linkedin",
        )

        related = analysis["related"]
        assert "synonyms" in related
        assert "complementary" in related