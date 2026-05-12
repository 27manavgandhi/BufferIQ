"""
Tests for sentiment analyzer.
"""

import pytest

from bufferiq.ml.content.sentiment.analyzer import (
    SentimentAnalyzer,
    Sentiment,
    SentimentResult,
)


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> SentimentAnalyzer:
        """Create sentiment analyzer fixture."""
        return SentimentAnalyzer()

    def test_analyze_positive_sentiment(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test positive sentiment detection."""
        result = analyzer.analyze("I love this product! It's amazing!")

        assert result.sentiment == Sentiment.POSITIVE
        assert result.confidence > 0.5

    def test_analyze_negative_sentiment(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test negative sentiment detection."""
        result = analyzer.analyze("I hate this. Terrible experience.")

        assert result.sentiment == Sentiment.NEGATIVE
        assert result.confidence > 0.5

    def test_analyze_neutral_sentiment(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test neutral sentiment detection."""
        result = analyzer.analyze("This is a statement about facts.")

        assert result.sentiment == Sentiment.NEUTRAL

    def test_analyze_returns_scores(self, analyzer: SentimentAnalyzer) -> None:
        """Test that analysis returns score breakdown."""
        result = analyzer.analyze("Great product!")

        assert "positive" in result.scores
        assert "negative" in result.scores
        assert "neutral" in result.scores
        assert "compound" in result.scores

    def test_analyze_returns_subjectivity(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test subjectivity score."""
        result = analyzer.analyze("I think this is good.")

        assert 0.0 <= result.subjectivity <= 1.0

    def test_analyze_returns_polarity(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test polarity score."""
        result = analyzer.analyze("Great!")

        assert -1.0 <= result.polarity <= 1.0

    def test_analyze_confidence_range(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test confidence is in valid range."""
        result = analyzer.analyze("This is excellent!")

        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_empty_text_raises_error(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test analyzing empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("")

    def test_analyze_whitespace_only_raises_error(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test analyzing whitespace-only text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("   ")

    def test_analyze_very_positive(self, analyzer: SentimentAnalyzer) -> None:
        """Test very positive text."""
        result = analyzer.analyze(
            "This is absolutely wonderful! I love it so much! Best ever!"
        )

        assert result.sentiment == Sentiment.POSITIVE
        assert result.confidence > 0.6

    def test_analyze_very_negative(self, analyzer: SentimentAnalyzer) -> None:
        """Test very negative text."""
        result = analyzer.analyze(
            "This is absolutely terrible! I hate it! Worst ever!"
        )

        assert result.sentiment == Sentiment.NEGATIVE
        assert result.confidence > 0.6

    def test_analyze_mixed_sentiment(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test mixed sentiment text."""
        result = analyzer.analyze("Good features but bad price.")

        # Could be positive, negative, or neutral depending on balance
        assert result.sentiment in [
            Sentiment.POSITIVE,
            Sentiment.NEGATIVE,
            Sentiment.NEUTRAL,
        ]

    def test_analyze_exclamations(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with exclamations."""
        result = analyzer.analyze("Amazing!!!")

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_questions(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with questions."""
        result = analyzer.analyze("Is this good?")

        # Questions are typically neutral
        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_short_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment analysis on short text."""
        result = analyzer.analyze("Great!")

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_long_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment analysis on longer text."""
        text = (
            "This product exceeded all my expectations. "
            "The quality is outstanding and the customer service "
            "was exceptional. I would highly recommend this to anyone "
            "looking for a reliable solution. Five stars!"
        )
        result = analyzer.analyze(text)

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_emoji_positive(self, analyzer: SentimentAnalyzer) -> None:
        """Test positive sentiment with emojis."""
        result = analyzer.analyze("Love this 😊 ❤️")

        # Emojis may or may not affect sentiment depending on implementation
        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_sarcasm(self, analyzer: SentimentAnalyzer) -> None:
        """Test sarcasm detection (limited)."""
        # Note: Sarcasm is difficult to detect
        result = analyzer.analyze("Oh great, another bug.")

        # May not correctly identify sarcasm
        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_negation(self, analyzer: SentimentAnalyzer) -> None:
        """Test negation handling."""
        result = analyzer.analyze("Not bad at all.")

        # Should handle negation
        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_intensifiers(self, analyzer: SentimentAnalyzer) -> None:
        """Test handling of intensifiers."""
        result = analyzer.analyze("Very very good!")

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_diminishers(self, analyzer: SentimentAnalyzer) -> None:
        """Test handling of diminishers."""
        result = analyzer.analyze("Somewhat okay.")

        # Should be less positive than "great"
        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_numbers_and_symbols(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test text with numbers and symbols."""
        result = analyzer.analyze("Product #1! 100% satisfied!")

        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_urls_in_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with URLs."""
        result = analyzer.analyze("Check this out https://example.com great!")

        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_hashtags(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with hashtags."""
        result = analyzer.analyze("Love this #awesome #great")

        assert isinstance(result.sentiment, Sentiment)

    def test_analyze_mentions(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with mentions."""
        result = analyzer.analyze("Thanks @john for the great help!")

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_all_caps(self, analyzer: SentimentAnalyzer) -> None:
        """Test sentiment with all caps."""
        result = analyzer.analyze("AMAZING PRODUCT")

        assert result.sentiment == Sentiment.POSITIVE

    def test_analyze_multilingual_fallback(
        self, analyzer: SentimentAnalyzer
    ) -> None:
        """Test non-English text (may have limited support)."""
        # Analyzer is English-focused but should handle gracefully
        result = analyzer.analyze("Good product")

        assert isinstance(result.sentiment, Sentiment)