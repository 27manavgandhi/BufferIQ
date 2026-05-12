"""
Tests for content diversity analyzer.
"""

from datetime import datetime

import pytest

from bufferiq.ml.content.diversity.analyzer import (
    ContentDiversityAnalyzer,
    DiversityMetrics,
)


class TestContentDiversityAnalyzer:
    """Test ContentDiversityAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ContentDiversityAnalyzer:
        """Create analyzer fixture."""
        return ContentDiversityAnalyzer()

    @pytest.fixture
    def sample_posts(self) -> list:
        """Create sample posts."""
        return [
            {
                "text": "AI post #AI #ML",
                "platform": "linkedin",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "sentiment": "positive",
            },
            {
                "text": "Data science #Data",
                "platform": "twitter",
                "created_at": datetime(2024, 1, 1, 14, 0),
                "sentiment": "neutral",
            },
            {
                "text": "Machine learning #ML",
                "platform": "linkedin",
                "created_at": datetime(2024, 1, 2, 10, 0),
                "sentiment": "positive",
            },
        ]

    def test_analyze_basic(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test basic diversity analysis."""
        metrics = analyzer.analyze(sample_posts)

        assert isinstance(metrics, DiversityMetrics)

    def test_analyze_returns_all_metrics(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test analysis returns all metric types."""
        metrics = analyzer.analyze(sample_posts)

        assert 0.0 <= metrics.topic_diversity <= 1.0
        assert 0.0 <= metrics.temporal_diversity <= 1.0
        assert 0.0 <= metrics.platform_diversity <= 1.0
        assert 0.0 <= metrics.sentiment_diversity <= 1.0
        assert 0.0 <= metrics.repetition_score <= 1.0
        assert 0.0 <= metrics.overall_diversity <= 1.0

    def test_analyze_empty_posts_raises_error(
        self, analyzer: ContentDiversityAnalyzer
    ) -> None:
        """Test empty posts raises error."""
        with pytest.raises(ValueError, match="Posts list cannot be empty"):
            analyzer.analyze([])

    def test_analyze_high_diversity(
        self, analyzer: ContentDiversityAnalyzer
    ) -> None:
        """Test high diversity posts."""
        posts = [
            {
                "text": "AI topic #AI",
                "platform": "linkedin",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "sentiment": "positive",
            },
            {
                "text": "Data topic #Data",
                "platform": "twitter",
                "created_at": datetime(2024, 1, 1, 15, 0),
                "sentiment": "negative",
            },
            {
                "text": "Tech topic #Tech",
                "platform": "bluesky",
                "created_at": datetime(2024, 1, 2, 12, 0),
                "sentiment": "neutral",
            },
        ]
        metrics = analyzer.analyze(posts)

        # High diversity across dimensions
        assert metrics.overall_diversity > 0.3

    def test_analyze_low_diversity(
        self, analyzer: ContentDiversityAnalyzer
    ) -> None:
        """Test low diversity posts."""
        posts = [
            {
                "text": "Same content",
                "platform": "linkedin",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "sentiment": "positive",
            },
            {
                "text": "Same content",
                "platform": "linkedin",
                "created_at": datetime(2024, 1, 1, 9, 30),
                "sentiment": "positive",
            },
        ]
        metrics = analyzer.analyze(posts)

        # Low diversity
        assert metrics.repetition_score > 0.5

    def test_analyze_topic_diversity(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test topic diversity calculation."""
        metrics = analyzer.analyze(sample_posts)

        assert isinstance(metrics.topic_diversity, float)

    def test_analyze_temporal_diversity(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test temporal diversity calculation."""
        metrics = analyzer.analyze(sample_posts)

        assert isinstance(metrics.temporal_diversity, float)

    def test_analyze_platform_diversity(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test platform diversity calculation."""
        metrics = analyzer.analyze(sample_posts)

        assert isinstance(metrics.platform_diversity, float)

    def test_analyze_sentiment_diversity(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test sentiment diversity calculation."""
        metrics = analyzer.analyze(sample_posts)

        assert isinstance(metrics.sentiment_diversity, float)

    def test_analyze_repetition_score(
        self, analyzer: ContentDiversityAnalyzer
    ) -> None:
        """Test repetition score calculation."""
        posts = [
            {"text": "Same text", "created_at": datetime(2024, 1, 1, 9, 0)},
            {"text": "Same text", "created_at": datetime(2024, 1, 1, 10, 0)},
            {"text": "Different text", "created_at": datetime(2024, 1, 1, 11, 0)},
        ]
        metrics = analyzer.analyze(posts)

        # Should detect repetition
        assert metrics.repetition_score > 0.0

    def test_analyze_with_window_days(
        self, analyzer: ContentDiversityAnalyzer, sample_posts: list
    ) -> None:
        """Test analysis with custom window."""
        metrics = analyzer.analyze(sample_posts, window_days=7)

        assert isinstance(metrics, DiversityMetrics)