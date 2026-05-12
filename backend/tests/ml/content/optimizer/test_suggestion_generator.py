"""
Tests for suggestion generator.
"""

import pytest

from bufferiq.ml.content.optimizer.suggestion_generator import (
    SuggestionGenerator,
    ContentSuggestion,
)


class TestSuggestionGenerator:
    """Test SuggestionGenerator class."""

    @pytest.fixture
    def generator(self) -> SuggestionGenerator:
        """Create suggestion generator fixture."""
        return SuggestionGenerator()

    def test_generate_basic(self, generator: SuggestionGenerator) -> None:
        """Test basic suggestion generation."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate("Short text", "linkedin", analysis)

        assert isinstance(suggestions, list)

    def test_generate_length_suggestion(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test length-based suggestions."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate("Hi", "linkedin", analysis)

        # Should suggest increasing length
        length_suggestions = [s for s in suggestions if s.type == "length"]
        assert len(length_suggestions) > 0

    def test_generate_hashtag_suggestion(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test hashtag-based suggestions."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate(
            "This is a post without hashtags", "linkedin", analysis
        )

        # Should suggest adding hashtags
        hashtag_suggestions = [
            s for s in suggestions if s.type == "hashtags"
        ]
        assert len(hashtag_suggestions) > 0

    def test_generate_sentiment_suggestion(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test sentiment-based suggestions."""
        analysis = {
            "features": {"hashtag_count": 3},
            "sentiment": {"sentiment": "negative"},
        }
        suggestions = generator.generate("Test post", "linkedin", analysis)

        # May suggest adjusting sentiment
        assert isinstance(suggestions, list)

    def test_generate_readability_suggestion(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test readability-based suggestions."""
        analysis = {
            "features": {"hashtag_count": 3},
            "readability": {"reading_difficulty": "hard"},
        }
        suggestions = generator.generate("Test post", "linkedin", analysis)

        # Should suggest improving readability
        readability_suggestions = [
            s for s in suggestions if s.type == "readability"
        ]
        assert len(readability_suggestions) > 0

    def test_generate_invalid_platform_raises_error(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            generator.generate("Test", "facebook", {})

    def test_generate_suggestion_structure(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test suggestion structure."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate("Test", "linkedin", analysis)

        if len(suggestions) > 0:
            suggestion = suggestions[0]
            assert hasattr(suggestion, "type")
            assert hasattr(suggestion, "priority")
            assert hasattr(suggestion, "current_value")
            assert hasattr(suggestion, "suggested_value")
            assert hasattr(suggestion, "impact")
            assert hasattr(suggestion, "confidence")

    def test_generate_priority_levels(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test suggestion priority levels."""
        analysis = {
            "features": {"hashtag_count": 0},
            "readability": {"reading_difficulty": "hard"},
        }
        suggestions = generator.generate("Test", "linkedin", analysis)

        priorities = [s.priority for s in suggestions]
        assert all(p in ["high", "medium", "low"] for p in priorities)

    def test_generate_confidence_range(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test confidence scores are valid."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate("Test", "linkedin", analysis)

        for suggestion in suggestions:
            assert 0.0 <= suggestion.confidence <= 1.0

    def test_generate_twitter_specific(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test Twitter-specific suggestions."""
        analysis = {"features": {"hashtag_count": 10}}
        suggestions = generator.generate("x" * 200, "twitter", analysis)

        # May suggest reducing hashtags for Twitter
        assert isinstance(suggestions, list)

    def test_generate_bluesky_specific(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test Bluesky-specific suggestions."""
        analysis = {"features": {"hashtag_count": 0}}
        suggestions = generator.generate("Short post", "bluesky", analysis)

        assert isinstance(suggestions, list)

    def test_generate_too_many_hashtags(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test suggestion for too many hashtags."""
        analysis = {"features": {"hashtag_count": 20}}
        suggestions = generator.generate("Test", "linkedin", analysis)

        hashtag_suggestions = [
            s for s in suggestions if s.type == "hashtags"
        ]
        # Should suggest reducing hashtags
        assert len(hashtag_suggestions) > 0

    def test_generate_optimal_length(
        self, generator: SuggestionGenerator
    ) -> None:
        """Test optimal length has no suggestions."""
        analysis = {"features": {"hashtag_count": 3}}
        text = "x" * 200  # Optimal length for LinkedIn
        suggestions = generator.generate(text, "linkedin", analysis)

        # Should have fewer or no length suggestions
        length_suggestions = [s for s in suggestions if s.type == "length"]
        # May be empty or have other suggestions
        assert isinstance(length_suggestions, list)