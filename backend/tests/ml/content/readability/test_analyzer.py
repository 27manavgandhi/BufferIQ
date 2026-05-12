"""
Tests for readability analyzer.
"""

import pytest

from bufferiq.ml.content.readability.analyzer import (
    ReadabilityAnalyzer,
    ReadabilityScores,
)


class TestReadabilityAnalyzer:
    """Test ReadabilityAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ReadabilityAnalyzer:
        """Create readability analyzer fixture."""
        return ReadabilityAnalyzer()

    def test_analyze_simple_text(self, analyzer: ReadabilityAnalyzer) -> None:
        """Test analyzing simple text."""
        text = "The cat sat on the mat. The dog ran fast."
        scores = analyzer.analyze(text)

        assert isinstance(scores, ReadabilityScores)
        assert scores.flesch_reading_ease > 0
        assert scores.average_grade_level >= 0

    def test_analyze_complex_text(self, analyzer: ReadabilityAnalyzer) -> None:
        """Test analyzing complex text."""
        text = (
            "The implementation of sophisticated algorithms necessitates "
            "comprehensive understanding of mathematical principles and "
            "computational complexity theory."
        )
        scores = analyzer.analyze(text)

        assert isinstance(scores, ReadabilityScores)
        assert scores.reading_difficulty in ["easy", "medium", "hard"]

    def test_analyze_returns_all_scores(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test that analysis returns all score types."""
        text = "This is a test sentence for readability analysis."
        scores = analyzer.analyze(text)

        assert hasattr(scores, "flesch_reading_ease")
        assert hasattr(scores, "flesch_kincaid_grade")
        assert hasattr(scores, "gunning_fog")
        assert hasattr(scores, "smog_index")
        assert hasattr(scores, "coleman_liau")
        assert hasattr(scores, "average_grade_level")
        assert hasattr(scores, "reading_difficulty")

    def test_analyze_calculates_average(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test average grade level calculation."""
        text = "The quick brown fox jumps over the lazy dog."
        scores = analyzer.analyze(text)

        assert scores.average_grade_level >= 0

    def test_analyze_difficulty_easy(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test easy difficulty classification."""
        text = "The cat sat. The dog ran. The bird flew."
        scores = analyzer.analyze(text)

        assert scores.reading_difficulty == "easy"

    def test_analyze_too_short_raises_error(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test analyzing too short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("Hi")

    def test_get_difficulty_level_easy(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test difficulty level classification for easy."""
        difficulty = analyzer.get_difficulty_level(4.5)
        assert difficulty == "easy"

    def test_get_difficulty_level_medium(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test difficulty level classification for medium."""
        difficulty = analyzer.get_difficulty_level(8.0)
        assert difficulty == "medium"

    def test_get_difficulty_level_hard(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test difficulty level classification for hard."""
        difficulty = analyzer.get_difficulty_level(14.0)
        assert difficulty == "hard"

    def test_analyze_scores_in_valid_ranges(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test that all scores are in valid ranges."""
        text = "This is a medium-length sentence for testing purposes."
        scores = analyzer.analyze(text)

        assert 0.0 <= scores.flesch_reading_ease <= 100.0
        assert scores.flesch_kincaid_grade >= 0
        assert scores.gunning_fog >= 0
        assert scores.smog_index >= 0
        assert scores.coleman_liau >= 0

    def test_analyze_long_text(self, analyzer: ReadabilityAnalyzer) -> None:
        """Test analyzing longer text."""
        text = (
            "This is a longer piece of text that should be analyzed properly. "
            "It contains multiple sentences with varying complexity. "
            "Some sentences are simple. Others are more complex and demonstrate "
            "the ability to handle different writing styles and patterns."
        )
        scores = analyzer.analyze(text)

        assert isinstance(scores, ReadabilityScores)

    def test_analyze_with_punctuation(
        self, analyzer: ReadabilityAnalyzer
    ) -> None:
        """Test analyzing text with various punctuation."""
        text = "Hello! How are you? I'm fine. Great day, isn't it?"
        scores = analyzer.analyze(text)

        assert isinstance(scores, ReadabilityScores)