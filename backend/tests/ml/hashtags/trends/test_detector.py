"""Tests for trend detector."""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from bufferiq.ml.hashtags.trends.detector import (
    TrendDetector,
    TrendingHashtag,
    TrendStage,
)


class TestTrendDetector:
    """Test TrendDetector class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def detector(self, mock_db):
        """Create detector instance."""
        return TrendDetector(db_session=mock_db)

    @pytest.mark.asyncio
    async def test_detect_trending(self, detector):
        """Test trending detection."""
        trending = await detector.detect_trending(
            platform="linkedin",
            limit=20,
        )

        assert isinstance(trending, list)
        assert all(isinstance(t, TrendingHashtag) for t in trending)

    @pytest.mark.asyncio
    async def test_detect_trending_with_category(self, detector):
        """Test trending with category filter."""
        trending = await detector.detect_trending(
            platform="linkedin",
            category="technology",
            limit=20,
        )

        assert isinstance(trending, list)

    @pytest.mark.asyncio
    async def test_detect_trending_invalid_platform(self, detector):
        """Test with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await detector.detect_trending(
                platform="facebook",
            )

    @pytest.mark.asyncio
    async def test_trending_hashtag_structure(self, detector):
        """Test trending hashtag structure."""
        trending = await detector.detect_trending(
            platform="linkedin",
            limit=1,
        )

        if trending:
            hashtag = trending[0]
            assert hasattr(hashtag, "hashtag")
            assert hasattr(hashtag, "platform")
            assert hasattr(hashtag, "stage")
            assert hasattr(hashtag, "momentum_score")
            assert hasattr(hashtag, "current_volume")

    @pytest.mark.asyncio
    async def test_trending_stages(self, detector):
        """Test trending stages."""
        trending = await detector.detect_trending(
            platform="linkedin",
            limit=20,
        )

        for t in trending:
            assert isinstance(t.stage, TrendStage)
            assert t.stage in [
                TrendStage.EMERGING,
                TrendStage.RISING,
                TrendStage.PEAK,
                TrendStage.DECLINING,
                TrendStage.DORMANT,
            ]

    def test_calculate_momentum(self, detector):
        """Test momentum calculation."""
        momentum = detector.calculate_momentum(
            current_volume=1000,
            previous_volume=500,
            velocity=0.8,
        )

        assert 0 <= momentum <= 100
        assert momentum > 50  # Should be high for doubling

    def test_calculate_momentum_zero_previous(self, detector):
        """Test momentum with zero previous volume."""
        momentum = detector.calculate_momentum(
            current_volume=1000,
            previous_volume=0,
            velocity=0.5,
        )

        assert 0 <= momentum <= 100

    @pytest.mark.asyncio
    async def test_trending_recommendation(self, detector):
        """Test recommendation field."""
        trending = await detector.detect_trending(
            platform="linkedin",
            limit=20,
        )

        for t in trending:
            assert t.recommendation in ["use_now", "monitor", "avoid"]

    @pytest.mark.asyncio
    async def test_trending_sorted_by_momentum(self, detector):
        """Test results sorted by momentum."""
        trending = await detector.detect_trending(
            platform="linkedin",
            limit=20,
        )

        if len(trending) > 1:
            # Should be sorted descending
            momentum_scores = [t.momentum_score for t in trending]
            assert momentum_scores == sorted(momentum_scores, reverse=True)


class TestMomentumScorer:
    """Test MomentumScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        from bufferiq.ml.hashtags.trends.momentum_scorer import MomentumScorer
        return MomentumScorer()

    def test_calculate(self, scorer):
        """Test momentum calculation."""
        momentum = scorer.calculate(
            current_volume=1500,
            previous_volume=1000,
            velocity=0.5,
        )

        assert 0 <= momentum <= 100
        assert momentum > 0

    def test_calculate_with_weights(self, scorer):
        """Test with custom weights."""
        momentum = scorer.calculate(
            current_volume=1500,
            previous_volume=1000,
            velocity=0.5,
            volume_weight=0.7,
            velocity_weight=0.3,
        )

        assert 0 <= momentum <= 100

    def test_calculate_batch(self, scorer):
        """Test batch calculation."""
        volume_pairs = [(1500, 1000), (2000, 1500), (1800, 1600)]
        velocities = [0.5, 0.6, 0.4]

        scores = scorer.calculate_batch(volume_pairs, velocities)

        assert len(scores) == 3
        assert all(0 <= s <= 100 for s in scores)