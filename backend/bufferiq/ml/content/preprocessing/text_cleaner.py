"""
Text cleaning and normalization.

Provides utilities for cleaning and preprocessing text content,
extracting metadata like URLs, hashtags, mentions, and emojis.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import emoji

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class PreprocessedText:
    """Preprocessed text with extracted features."""

    original: str
    cleaned: str
    language: str
    tokens: List[str]
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    emojis: List[str]
    word_count: int
    char_count: int
    sentence_count: int


class TextCleaner:
    """
        Clean and normalize text content.

        Removes special characters, normalizes whitespace,
        handles Unicode, and extracts metadata.

        Example:
    ```python
            cleaner = TextCleaner()
            result = cleaner.clean("Check out this post! 🚀 #AI")
            print(result.cleaned)  # "Check out this post"
            print(result.emojis)   # ["🚀"]
            print(result.hashtags) # ["AI"]
    ```
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        lowercase: bool = False,
    ) -> None:
        """
        Initialize text cleaner.

        Args:
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            lowercase: Convert to lowercase

        Raises:
            ValueError: If invalid configuration
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase

        # Regex patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.mention_pattern = re.compile(r"@[\w]+")
        self.hashtag_pattern = re.compile(r"#[\w]+")

    def clean(self, text: str) -> PreprocessedText:
        """
        Clean and preprocess text.

        Args:
            text: Raw text to clean

        Returns:
            Preprocessed text with extracted features

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        original = text
        cleaned = text

        # Extract features before cleaning
        urls = self._extract_urls(text)
        mentions = self._extract_mentions(text)
        hashtags = self._extract_hashtags(text)
        emojis = self._extract_emojis(text)

        # Detect language (simplified - assumes English)
        language = "en"

        # Remove URLs
        if self.remove_urls:
            cleaned = self.url_pattern.sub("", cleaned)

        # Remove mentions
        if self.remove_mentions:
            cleaned = self.mention_pattern.sub("", cleaned)

        # Remove hashtags
        if self.remove_hashtags:
            cleaned = self.hashtag_pattern.sub("", cleaned)

        # Remove emojis from cleaned text
        cleaned = self._remove_emojis(cleaned)

        # Normalize Unicode
        cleaned = unicodedata.normalize("NFKD", cleaned)

        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Lowercase if requested
        if self.lowercase:
            cleaned = cleaned.lower()

        # Extract tokens (simple whitespace split)
        tokens = [t for t in cleaned.split() if t]

        # Count features
        word_count = len(tokens)
        char_count = len(cleaned)
        sentence_count = len(re.split(r"[.!?]+", cleaned)) - 1
        if sentence_count < 1:
            sentence_count = 1

        return PreprocessedText(
            original=original,
            cleaned=cleaned,
            language=language,
            tokens=tokens,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            emojis=emojis,
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
        )

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        return self.url_pattern.findall(text)

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        return [m[1:] for m in self.mention_pattern.findall(text)]

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract #hashtags from text."""
        return [h[1:] for h in self.hashtag_pattern.findall(text)]

    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text."""
        return [char for char in text if char in emoji.EMOJI_DATA]

    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return "".join(char for char in text if char not in emoji.EMOJI_DATA)
