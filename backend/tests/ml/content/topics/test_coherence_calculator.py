"""
Tests for coherence calculator.
"""

import pytest

from bufferiq.ml.content.topics.coherence_calculator import CoherenceCalculator


class TestCoherenceCalculator:
    """Test CoherenceCalculator class."""

    @pytest.fixture
    def calculator(self) -> CoherenceCalculator:
        """Create coherence calculator fixture."""
        return CoherenceCalculator()

    @pytest.fixture
    def sample_documents(self) -> list:
        """Create sample documents for testing."""
        return [
            "machine learning artificial intelligence",
            "machine learning data science",
            "artificial intelligence neural networks",
            "data science analytics",
        ]

    def test_calculate_basic(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test basic coherence calculation."""
        keywords = ["machine", "learning"]
        coherence = calculator.calculate(keywords, sample_documents)

        assert 0.0 <= coherence <= 1.0

    def test_calculate_high_coherence(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation with high coherence keywords."""
        keywords = ["machine", "learning", "data"]
        coherence = calculator.calculate(keywords, sample_documents)

        # These words appear together, should have some coherence
        assert coherence > 0.0

    def test_calculate_low_coherence(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation with low coherence keywords."""
        keywords = ["xyz", "abc", "def"]
        coherence = calculator.calculate(keywords, sample_documents)

        # These words don't appear, should have zero coherence
        assert coherence == 0.0

    def test_calculate_single_keyword(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation with single keyword."""
        keywords = ["machine"]
        coherence = calculator.calculate(keywords, sample_documents)

        # Single keyword has no co-occurrence
        assert coherence == 0.0

    def test_calculate_empty_keywords_raises_error(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test empty keywords raises error."""
        with pytest.raises(ValueError, match="Keywords list cannot be empty"):
            calculator.calculate([], sample_documents)

    def test_calculate_empty_documents_raises_error(
        self, calculator: CoherenceCalculator
    ) -> None:
        """Test empty documents raises error."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            calculator.calculate(["machine", "learning"], [])

    def test_calculate_two_keywords(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation with two keywords."""
        keywords = ["machine", "learning"]
        coherence = calculator.calculate(keywords, sample_documents)

        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0

    def test_calculate_many_keywords(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation with many keywords."""
        keywords = ["machine", "learning", "data", "science", "artificial"]
        coherence = calculator.calculate(keywords, sample_documents)

        assert 0.0 <= coherence <= 1.0

    def test_calculate_case_insensitive(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test calculation is case-insensitive."""
        keywords = ["MACHINE", "LEARNING"]
        coherence = calculator.calculate(keywords, sample_documents)

        assert coherence > 0.0

    def test_calculate_umass_basic(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test UMass coherence calculation."""
        keywords = ["machine", "learning"]
        coherence = calculator.calculate_umass(keywords, sample_documents)

        assert isinstance(coherence, float)

    def test_calculate_umass_empty_keywords_raises_error(
        self, calculator: CoherenceCalculator, sample_documents: list
    ) -> None:
        """Test UMass with empty keywords raises error."""
        with pytest.raises(ValueError, match="Keywords list cannot be empty"):
            calculator.calculate_umass([], sample_documents)

    def test_calculate_umass_empty_documents_raises_error(
        self, calculator: CoherenceCalculator
    ) -> None:
        """Test UMass with empty documents raises error."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            calculator.calculate_umass(["machine", "learning"], [])

    def test_calculate_perfect_coherence(
        self, calculator: CoherenceCalculator
    ) -> None:
        """Test perfect coherence scenario."""
        docs = ["word1 word2", "word1 word2", "word1 word2"]
        keywords = ["word1", "word2"]
        coherence = calculator.calculate(keywords, docs)

        # Perfect co-occurrence
        assert coherence == 1.0

    def test_calculate_no_cooccurrence(
        self, calculator: CoherenceCalculator
    ) -> None:
        """Test no co-occurrence scenario."""
        docs = ["word1 only", "word2 only", "word3 only"]
        keywords = ["word1", "word2"]
        coherence = calculator.calculate(keywords, docs)

        # No co-occurrence
        assert coherence == 0.0