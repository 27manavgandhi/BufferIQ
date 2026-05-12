"""
Text feature extraction.

Extracts various features from preprocessed text for ML models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from bufferiq.ml.content.preprocessing.text_cleaner import PreprocessedText


@dataclass
class TextFeatures:
    """Extracted text features."""

    # Basic counts
    word_count: int
    char_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float

    # Content features
    has_url: bool
    url_count: int
    has_hashtag: bool
    hashtag_count: int
    has_mention: bool
    mention_count: int
    has_emoji: bool
    emoji_count: int

    # Punctuation
    has_question: bool
    has_exclamation: bool
    question_count: int
    exclamation_count: int

    # Capitalization
    uppercase_ratio: float
    capitalized_word_ratio: float


class TextFeatureExtractor:
    """
        Extract features from preprocessed text.

        Example:
    ```python
            cleaner = TextCleaner()
            extractor = TextFeatureExtractor()

            preprocessed = cleaner.clean("Check this out! 🚀 #AI")
            features = extractor.extract(preprocessed)
            print(features.has_emoji)  # True
            print(features.has_hashtag)  # True
    ```
    """

    def extract(self, preprocessed: PreprocessedText) -> TextFeatures:
        """
        Extract features from preprocessed text.

        Args:
            preprocessed: Preprocessed text

        Returns:
            Extracted features

        Raises:
            ValueError: If preprocessed text is invalid
        """
        if not preprocessed.cleaned:
            raise ValueError("Preprocessed text cannot be empty")

        # Basic counts
        word_count = preprocessed.word_count
        char_count = preprocessed.char_count
        sentence_count = preprocessed.sentence_count

        # Average lengths
        avg_word_length = char_count / word_count if word_count > 0 else 0.0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

        # Content features
        has_url = len(preprocessed.urls) > 0
        url_count = len(preprocessed.urls)
        has_hashtag = len(preprocessed.hashtags) > 0
        hashtag_count = len(preprocessed.hashtags)
        has_mention = len(preprocessed.mentions) > 0
        mention_count = len(preprocessed.mentions)
        has_emoji = len(preprocessed.emojis) > 0
        emoji_count = len(preprocessed.emojis)

        # Punctuation
        has_question = "?" in preprocessed.original
        has_exclamation = "!" in preprocessed.original
        question_count = preprocessed.original.count("?")
        exclamation_count = preprocessed.original.count("!")

        # Capitalization
        uppercase_ratio = self._calculate_uppercase_ratio(preprocessed.original)
        capitalized_word_ratio = self._calculate_capitalized_ratio(preprocessed.tokens)

        return TextFeatures(
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            has_url=has_url,
            url_count=url_count,
            has_hashtag=has_hashtag,
            hashtag_count=hashtag_count,
            has_mention=has_mention,
            mention_count=mention_count,
            has_emoji=has_emoji,
            emoji_count=emoji_count,
            has_question=has_question,
            has_exclamation=has_exclamation,
            question_count=question_count,
            exclamation_count=exclamation_count,
            uppercase_ratio=uppercase_ratio,
            capitalized_word_ratio=capitalized_word_ratio,
        )

    def _calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters."""
        if not text:
            return 0.0
        uppercase_count = sum(1 for c in text if c.isupper())
        alpha_count = sum(1 for c in text if c.isalpha())
        return uppercase_count / alpha_count if alpha_count > 0 else 0.0

    def _calculate_capitalized_ratio(self, tokens: List[str]) -> float:
        """Calculate ratio of capitalized words."""
        if not tokens:
            return 0.0
        capitalized_count = sum(1 for t in tokens if t and t[0].isupper())
        return capitalized_count / len(tokens)
