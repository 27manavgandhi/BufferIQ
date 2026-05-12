"""
Tests for text cleaner.
"""

import pytest

from bufferiq.ml.content.preprocessing.text_cleaner import (
    TextCleaner,
    PreprocessedText,
)


class TestTextCleaner:
    """Test TextCleaner class."""

    def test_clean_simple_text(self) -> None:
        """Test cleaning simple text."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello world!")

        assert result.original == "Hello world!"
        assert result.cleaned == "Hello world!"
        assert result.word_count == 2
        assert result.sentence_count == 1

    def test_clean_with_urls(self) -> None:
        """Test URL extraction and removal."""
        cleaner = TextCleaner(remove_urls=True)
        result = cleaner.clean("Check https://example.com out!")

        assert "https://" not in result.cleaned
        assert len(result.urls) == 1
        assert result.urls[0] == "https://example.com"

    def test_clean_with_hashtags(self) -> None:
        """Test hashtag extraction."""
        cleaner = TextCleaner(remove_hashtags=False)
        result = cleaner.clean("Great post! #AI #MachineLearning")

        assert len(result.hashtags) == 2
        assert "AI" in result.hashtags
        assert "MachineLearning" in result.hashtags

    def test_clean_with_mentions(self) -> None:
        """Test mention extraction."""
        cleaner = TextCleaner(remove_mentions=False)
        result = cleaner.clean("Thanks @john and @jane!")

        assert len(result.mentions) == 2
        assert "john" in result.mentions
        assert "jane" in result.mentions

    def test_clean_with_emojis(self) -> None:
        """Test emoji extraction."""
        cleaner = TextCleaner()
        result = cleaner.clean("Great work! 🚀 💯")

        assert len(result.emojis) >= 2
        assert "🚀" in result.emojis
        assert "💯" in result.emojis
        # Emojis should be removed from cleaned text
        assert "🚀" not in result.cleaned

    def test_lowercase_option(self) -> None:
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        result = cleaner.clean("Hello WORLD")

        assert result.cleaned == "hello world"

    def test_remove_mentions_option(self) -> None:
        """Test mention removal."""
        cleaner = TextCleaner(remove_mentions=True)
        result = cleaner.clean("Hi @john how are you?")

        assert "@john" not in result.cleaned
        assert len(result.mentions) == 1

    def test_remove_hashtags_option(self) -> None:
        """Test hashtag removal."""
        cleaner = TextCleaner(remove_hashtags=True)
        result = cleaner.clean("Check this #awesome post")

        assert "#awesome" not in result.cleaned
        assert len(result.hashtags) == 1

    def test_clean_empty_text(self) -> None:
        """Test cleaning empty text raises error."""
        cleaner = TextCleaner()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            cleaner.clean("")

    def test_clean_whitespace_only(self) -> None:
        """Test cleaning whitespace-only text raises error."""
        cleaner = TextCleaner()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            cleaner.clean("   ")

    def test_word_count(self) -> None:
        """Test word count calculation."""
        cleaner = TextCleaner()
        result = cleaner.clean("This is a test sentence.")

        assert result.word_count == 5

    def test_char_count(self) -> None:
        """Test character count."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello")

        assert result.char_count == 5

    def test_sentence_count(self) -> None:
        """Test sentence count."""
        cleaner = TextCleaner()
        result = cleaner.clean("First sentence. Second sentence! Third?")

        assert result.sentence_count >= 3

    def test_multiple_urls(self) -> None:
        """Test multiple URL extraction."""
        cleaner = TextCleaner()
        result = cleaner.clean(
            "Visit https://example.com and http://test.com"
        )

        assert len(result.urls) == 2

    def test_unicode_normalization(self) -> None:
        """Test Unicode normalization."""
        cleaner = TextCleaner()
        result = cleaner.clean("café résumé")

        assert isinstance(result.cleaned, str)

    def test_extra_whitespace_removal(self) -> None:
        """Test extra whitespace removal."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello    world   test")

        assert "    " not in result.cleaned
        assert result.cleaned == "Hello world test"

    def test_language_detection(self) -> None:
        """Test language detection (simplified)."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello world")

        assert result.language == "en"

    def test_tokens_extraction(self) -> None:
        """Test token extraction."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello world test")

        assert len(result.tokens) == 3
        assert "Hello" in result.tokens or "hello" in result.tokens

    def test_complex_text(self) -> None:
        """Test complex text with multiple features."""
        cleaner = TextCleaner()
        text = (
            "Check out https://example.com! "
            "Great work @john #AI #ML 🚀 "
            "More info at http://test.com"
        )
        result = cleaner.clean(text)

        assert len(result.urls) == 2
        assert len(result.mentions) == 1
        assert len(result.hashtags) == 2
        assert len(result.emojis) >= 1

    def test_no_urls_found(self) -> None:
        """Test text with no URLs."""
        cleaner = TextCleaner()
        result = cleaner.clean("Simple text without links")

        assert len(result.urls) == 0

    def test_no_hashtags_found(self) -> None:
        """Test text with no hashtags."""
        cleaner = TextCleaner()
        result = cleaner.clean("Simple text without hashtags")

        assert len(result.hashtags) == 0

    def test_no_mentions_found(self) -> None:
        """Test text with no mentions."""
        cleaner = TextCleaner()
        result = cleaner.clean("Simple text without mentions")

        assert len(result.mentions) == 0

    def test_no_emojis_found(self) -> None:
        """Test text with no emojis."""
        cleaner = TextCleaner()
        result = cleaner.clean("Simple text without emojis")

        assert len(result.emojis) == 0