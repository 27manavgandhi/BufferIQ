"""
Tests for readability metrics.
"""

import pytest

from bufferiq.ml.content.readability.metrics import ReadabilityMetrics


class TestReadabilityMetrics:
    """Test ReadabilityMetrics class."""

    @pytest.fixture
    def metrics(self) -> ReadabilityMetrics:
        """Create readability metrics fixture."""
        return ReadabilityMetrics()

    def test_flesch_reading_ease_simple(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test Flesch Reading Ease on simple text."""
        text = "The cat sat on the mat."
        score = metrics.flesch_reading_ease(text)

        assert 0.0 <= score <= 100.0
        assert score > 80  # Simple text should be easy

    def test_flesch_reading_ease_complex(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test Flesch Reading Ease on complex text."""
        text = (
            "The implementation of sophisticated algorithms "
            "necessitates comprehensive understanding of mathematical principles."
        )
        score = metrics.flesch_reading_ease(text)

        assert 0.0 <= score <= 100.0

    def test_flesch_kincaid_grade_simple(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test Flesch-Kincaid Grade on simple text."""
        text = "The cat sat on the mat."
        grade = metrics.flesch_kincaid_grade(text)

        assert grade >= 0.0
        assert grade < 5  # Simple text should be low grade

    def test_flesch_kincaid_grade_complex(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test Flesch-Kincaid Grade on complex text."""
        text = (
            "The implementation of sophisticated algorithms "
            "necessitates comprehensive understanding."
        )
        grade = metrics.flesch_kincaid_grade(text)

        assert grade > 10  # Complex text should be higher grade

    def test_gunning_fog_index(self, metrics: ReadabilityMetrics) -> None:
        """Test Gunning Fog Index."""
        text = "The quick brown fox jumps over the lazy dog."
        fog = metrics.gunning_fog_index(text)

        assert fog >= 0.0

    def test_smog_index(self, metrics: ReadabilityMetrics) -> None:
        """Test SMOG Index."""
        text = "The cat sat on the mat. The dog ran in the park."
        smog = metrics.smog_index(text)

        assert smog >= 0.0

    def test_coleman_liau_index(self, metrics: ReadabilityMetrics) -> None:
        """Test Coleman-Liau Index."""
        text = "The cat sat on the mat."
        cli = metrics.coleman_liau_index(text)

        assert cli >= 0.0

    def test_count_words(self, metrics: ReadabilityMetrics) -> None:
        """Test word counting."""
        text = "Hello world test example"
        count = metrics._count_words(text)

        assert count == 4

    def test_count_sentences(self, metrics: ReadabilityMetrics) -> None:
        """Test sentence counting."""
        text = "First sentence. Second sentence! Third?"
        count = metrics._count_sentences(text)

        assert count >= 3

    def test_count_characters(self, metrics: ReadabilityMetrics) -> None:
        """Test character counting."""
        text = "Hello123"
        count = metrics._count_characters(text)

        assert count == 8

    def test_count_syllables(self, metrics: ReadabilityMetrics) -> None:
        """Test syllable counting."""
        text = "hello"
        count = metrics._count_syllables(text)

        assert count >= 2

    def test_syllables_in_word_simple(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test syllables in simple word."""
        syllables = metrics._syllables_in_word("cat")
        assert syllables >= 1

    def test_syllables_in_word_complex(
        self, metrics: ReadabilityMetrics
    ) -> None:
        """Test syllables in complex word."""
        syllables = metrics._syllables_in_word("beautiful")
        assert syllables >= 3

    def test_count_complex_words(self, metrics: ReadabilityMetrics) -> None:
        """Test complex word counting."""
        text = "The beautiful catastrophic implementation"
        count = metrics._count_complex_words(text)

        assert count > 0

    def test_empty_text_handling(self, metrics: ReadabilityMetrics) -> None:
        """Test handling of empty text."""
        score = metrics.flesch_reading_ease("")
        assert score == 0.0

    def test_single_word(self, metrics: ReadabilityMetrics) -> None:
        """Test metrics on single word."""
        text = "Hello"
        score = metrics.flesch_reading_ease(text)

        assert isinstance(score, float)

    def test_long_text(self, metrics: ReadabilityMetrics) -> None:
        """Test metrics on longer text."""
        text = (
            "This is a longer piece of text that contains multiple sentences. "
            "It should be analyzed properly by the readability metrics. "
            "The results should be meaningful and within expected ranges."
        )
        score = metrics.flesch_reading_ease(text)

        assert 0.0 <= score <= 100.0