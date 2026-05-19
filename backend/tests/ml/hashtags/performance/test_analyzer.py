"""Tests for hashtag performance analyzer."""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from bufferiq.ml.hashtags.performance.analyzer import (
    HashtagPerformanceAnalyzer,
    HashtagPerformance,
    HashtagABTest,
)


class TestHashtagPerformanceAnalyzer:
    """Test HashtagPerformanceAnalyzer class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create analyzer instance."""
        return HashtagPerformanceAnalyzer(db_session=mock_db)

    @pytest.mark.asyncio
    async def test_analyze_basic(self, analyzer):
        """Test basic hashtag analysis."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(performance, HashtagPerformance)
        assert performance.hashtag == "ai"
        assert performance.platform == "linkedin"
        assert performance.total_uses >= 0
        assert performance.avg_engagement >= 0

    @pytest.mark.asyncio
    async def test_analyze_with_user(self, analyzer):
        """Test analysis with user filter."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
            user_id="user123",
        )

        assert isinstance(performance, HashtagPerformance)
        assert performance.hashtag == "ai"

    @pytest.mark.asyncio
    async def test_analyze_invalid_platform(self, analyzer):
        """Test analysis with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await analyzer.analyze(
                hashtag="ai",
                platform="facebook",
            )

    @pytest.mark.asyncio
    async def test_analyze_engagement_metrics(self, analyzer):
        """Test engagement metrics calculation."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert hasattr(performance, "avg_engagement")
        assert hasattr(performance, "median_engagement")
        assert hasattr(performance, "total_engagement")
        assert hasattr(performance, "engagement_rate")

    @pytest.mark.asyncio
    async def test_analyze_trend_direction(self, analyzer):
        """Test trend direction detection."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert performance.trend_direction in [
            "growing",
            "stable",
            "declining",
        ]

    @pytest.mark.asyncio
    async def test_analyze_roi(self, analyzer):
        """Test ROI calculation."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert hasattr(performance, "estimated_roi")
        assert isinstance(performance.estimated_roi, float)

    @pytest.mark.asyncio
    async def test_compare_with_without(self, analyzer):
        """Test A/B comparison."""
        ab_test = await analyzer.compare_with_without(
            hashtag="ai",
            platform="linkedin",
            user_id="user123",
        )

        assert isinstance(ab_test, HashtagABTest)
        assert ab_test.hashtag == "ai"
        assert ab_test.platform == "linkedin"
        assert isinstance(ab_test.is_significant, bool)

    @pytest.mark.asyncio
    async def test_ab_test_statistics(self, analyzer):
        """Test A/B test statistics."""
        ab_test = await analyzer.compare_with_without(
            hashtag="ai",
            platform="linkedin",
            user_id="user123",
        )

        assert hasattr(ab_test, "t_statistic")
        assert hasattr(ab_test, "p_value")
        assert hasattr(ab_test, "effect_size")
        assert 0 <= ab_test.p_value <= 1

    @pytest.mark.asyncio
    async def test_ab_test_recommendation(self, analyzer):
        """Test A/B test recommendation."""
        ab_test = await analyzer.compare_with_without(
            hashtag="ai",
            platform="linkedin",
            user_id="user123",
        )

        assert hasattr(ab_test, "recommendation")
        assert isinstance(ab_test.recommendation, str)

    def test_calculate_roi(self, analyzer):
        """Test ROI calculation."""
        roi = analyzer.calculate_roi(
            avg_engagement_with=150.0,
            avg_engagement_without=120.0,
            hashtag_length=2,  # "ai"
        )

        # ROI = (150 - 120) / (2 + 1) = 30 / 3 = 10
        assert roi == 10.0

    def test_calculate_roi_zero_length(self, analyzer):
        """Test ROI with zero length."""
        roi = analyzer.calculate_roi(
            avg_engagement_with=150.0,
            avg_engagement_without=120.0,
            hashtag_length=0,
        )

        assert roi == 0.0

    @pytest.mark.asyncio
    async def test_analyze_lookback_days(self, analyzer):
        """Test analysis with different lookback periods."""
        performance_90 = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
            lookback_days=90,
        )

        performance_30 = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
            lookback_days=30,
        )

        assert isinstance(performance_90, HashtagPerformance)
        assert isinstance(performance_30, HashtagPerformance)

    @pytest.mark.asyncio
    async def test_analyze_percentiles(self, analyzer):
        """Test engagement percentiles."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(performance.engagement_percentiles, dict)
        assert 25 in performance.engagement_percentiles
        assert 50 in performance.engagement_percentiles
        assert 75 in performance.engagement_percentiles
        assert 90 in performance.engagement_percentiles

    @pytest.mark.asyncio
    async def test_analyze_dates(self, analyzer):
        """Test first/last used dates."""
        performance = await analyzer.analyze(
            hashtag="ai",
            platform="linkedin",
        )

        assert isinstance(performance.first_used, datetime)
        assert isinstance(performance.last_used, datetime)
        assert performance.first_used <= performance.last_used