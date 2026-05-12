"""
Tests for text feature extractor.
"""

import pytest

from bufferiq.ml.content.preprocessing.text_cleaner import TextCleaner
from bufferiq.ml.content.preprocessing.feature_extractor import (
    TextFeatureExtractor,
    TextFeatures,
)


class TestTextFeatureExtractor:
    """Test TextFeatureExtractor class."""

    @pytest.fixture
    def cleaner(self) -> TextCleaner:
        """Create text cleaner fixture."""
        return TextCleaner()

    @pytest.fixture
    def extractor(self) -> TextFeatureExtractor:
        """Create feature extractor fixture."""
        return TextFeatureExtractor()

    def test_extract_basic_features(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test extracting basic features."""
        preprocessed = cleaner.clean("Hello world!")
        features = extractor.extract(preprocessed)

        assert features.word_count == 2
        assert features.char_count == 12
        assert features.sentence_count == 1

    def test_extract_with_urls(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test URL feature extraction."""
        preprocessed = cleaner.clean("Check https://example.com out!")
        features = extractor.extract(preprocessed)

        assert features.has_url is True
        assert features.url_count == 1

    def test_extract_with_hashtags(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test hashtag feature extraction."""
        preprocessed = cleaner.clean("Great post! #AI #ML")
        features = extractor.extract(preprocessed)

        assert features.has_hashtag is True
        assert features.hashtag_count == 2

    def test_extract_with_mentions(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test mention feature extraction."""
        preprocessed = cleaner.clean("Thanks @john!")
        features = extractor.extract(preprocessed)

        assert features.has_mention is True
        assert features.mention_count == 1

    def test_extract_with_emojis(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test emoji feature extraction."""
        preprocessed = cleaner.clean("Great! 🚀 💯")
        features = extractor.extract(preprocessed)

        assert features.has_emoji is True
        assert features.emoji_count >= 2

    def test_extract_punctuation_features(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test punctuation feature extraction."""
        preprocessed = cleaner.clean("Really? Yes! Awesome!")
        features = extractor.extract(preprocessed)

        assert features.has_question is True
        assert features.has_exclamation is True
        assert features.question_count >= 1
        assert features.exclamation_count >= 1

    def test_extract_average_word_length(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test average word length calculation."""
        preprocessed = cleaner.clean("Hi there friend")
        features = extractor.extract(preprocessed)

        assert features.avg_word_length > 0

    def test_extract_average_sentence_length(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test average sentence length calculation."""
        preprocessed = cleaner.clean("First sentence. Second sentence here.")
        features = extractor.extract(preprocessed)

        assert features.avg_sentence_length > 0

    def test_extract_uppercase_ratio(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test uppercase ratio calculation."""
        preprocessed = cleaner.clean("HELLO world")
        features = extractor.extract(preprocessed)

        assert 0 <= features.uppercase_ratio <= 1

    def test_extract_capitalized_word_ratio(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test capitalized word ratio."""
        preprocessed = cleaner.clean("Hello World Test")
        features = extractor.extract(preprocessed)

        assert 0 <= features.capitalized_word_ratio <= 1

    def test_extract_empty_text_raises_error(
        self, extractor: TextFeatureExtractor
    ) -> None:
        """Test extracting from empty text raises error."""
        from bufferiq.ml.content.preprocessing.text_cleaner import (
            PreprocessedText,
        )

        preprocessed = PreprocessedText(
            original="",
            cleaned="",
            language="en",
            tokens=[],
            hashtags=[],
            mentions=[],
            urls=[],
            emojis=[],
            word_count=0,
            char_count=0,
            sentence_count=0,
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor.extract(preprocessed)

    def test_extract_no_urls(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test features when no URLs present."""
        preprocessed = cleaner.clean("Simple text")
        features = extractor.extract(preprocessed)

        assert features.has_url is False
        assert features.url_count == 0

    def test_extract_no_hashtags(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test features when no hashtags present."""
        preprocessed = cleaner.clean("Simple text")
        features = extractor.extract(preprocessed)

        assert features.has_hashtag is False
        assert features.hashtag_count == 0

    def test_extract_no_mentions(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test features when no mentions present."""
        preprocessed = cleaner.clean("Simple text")
        features = extractor.extract(preprocessed)

        assert features.has_mention is False
        assert features.mention_count == 0

    def test_extract_no_emojis(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test features when no emojis present."""
        preprocessed = cleaner.clean("Simple text")
        features = extractor.extract(preprocessed)

        assert features.has_emoji is False
        assert features.emoji_count == 0

    def test_extract_no_punctuation(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test features when no special punctuation."""
        preprocessed = cleaner.clean("Simple text here")
        features = extractor.extract(preprocessed)

        assert features.has_question is False
        assert features.has_exclamation is False

    def test_extract_complex_text(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test feature extraction on complex text."""
        text = (
            "Check https://example.com! "
            "Great work @john #AI 🚀 "
            "Really? Yes!"
        )
        preprocessed = cleaner.clean(text)
        features = extractor.extract(preprocessed)

        assert features.has_url is True
        assert features.has_mention is True
        assert features.has_hashtag is True
        assert features.has_emoji is True
        assert features.has_question is True
        assert features.has_exclamation is True

    def test_extract_all_lowercase(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test extraction with all lowercase text."""
        preprocessed = cleaner.clean("all lowercase text here")
        features = extractor.extract(preprocessed)

        assert features.uppercase_ratio == 0.0

    def test_extract_all_uppercase(
        self, cleaner: TextCleaner, extractor: TextFeatureExtractor
    ) -> None:
        """Test extraction with all uppercase text."""
        preprocessed = cleaner.clean("ALL UPPERCASE TEXT")
        features = extractor.extract(preprocessed)

        assert features.uppercase_ratio == 1.0