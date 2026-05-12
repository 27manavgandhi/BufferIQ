"""
Tests for topic diversity analyzer.
"""

import pytest

from bufferiq.ml.content.diversity.topic_diversity import TopicDiversityAnalyzer


class TestTopicDiversityAnalyzer:
    """Test TopicDiversityAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> TopicDiversityAnalyzer:
        """Create analyzer fixture."""
        return TopicDiversityAnalyzer()

    def test_calculate_diversity_basic(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test basic diversity calculation."""
        topics = ["AI", "ML", "Data", "AI", "ML"]
        diversity = analyzer.calculate_diversity(topics)

        assert 0.0 <= diversity <= 1.0

    def test_calculate_diversity_high(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test high diversity."""
        topics = ["AI", "ML", "Data", "Science", "Tech"]
        diversity = analyzer.calculate_diversity(topics)

        # All unique topics = high diversity
        assert diversity > 0.8

    def test_calculate_diversity_low(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test low diversity."""
        topics = ["AI", "AI", "AI", "AI", "AI"]
        diversity = analyzer.calculate_diversity(topics)

        # All same topic = low diversity
        assert diversity == 0.0

    def test_calculate_diversity_empty_raises_error(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test empty topics raises error."""
        with pytest.raises(ValueError, match="Topics list cannot be empty"):
            analyzer.calculate_diversity([])

    def test_calculate_topic_distribution(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test topic distribution calculation."""
        topics = ["AI", "AI", "ML", "Data"]
        distribution = analyzer.calculate_topic_distribution(topics)

        assert "AI" in distribution
        assert "ML" in distribution
        assert "Data" in distribution
        assert abs(sum(distribution.values()) - 1.0) < 0.01  # Sum to 1

    def test_calculate_topic_distribution_empty_raises_error(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test empty topics raises error."""
        with pytest.raises(ValueError, match="Topics list cannot be empty"):
            analyzer.calculate_topic_distribution([])

    def test_calculate_gini_coefficient(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test Gini coefficient calculation."""
        topics = ["AI", "AI", "ML", "Data"]
        gini = analyzer.calculate_gini_coefficient(topics)

        assert 0.0 <= gini <= 1.0

    def test_calculate_gini_perfect_equality(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test Gini coefficient for perfect equality."""
        topics = ["AI", "ML", "Data"]
        gini = analyzer.calculate_gini_coefficient(topics)

        # All equal frequency = low Gini
        assert gini < 0.5

    def test_calculate_gini_perfect_inequality(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test Gini coefficient for inequality."""
        topics = ["AI"] * 100 + ["ML"]
        gini = analyzer.calculate_gini_coefficient(topics)

        # One dominant topic = higher Gini
        assert gini > 0.0

    def test_calculate_gini_empty_raises_error(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test empty topics raises error."""
        with pytest.raises(ValueError, match="Topics list cannot be empty"):
            analyzer.calculate_gini_coefficient([])

    def test_calculate_diversity_single_topic(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test diversity with single unique topic."""
        topics = ["AI"]
        diversity = analyzer.calculate_diversity(topics)

        # Single topic = no diversity
        assert diversity == 0.0

    def test_calculate_diversity_two_topics(
        self, analyzer: TopicDiversityAnalyzer
    ) -> None:
        """Test diversity with two topics."""
        topics = ["AI", "ML", "AI", "ML"]
        diversity = analyzer.calculate_diversity(topics)

        # Perfect split = maximum diversity for 2 topics
        assert diversity > 0.9