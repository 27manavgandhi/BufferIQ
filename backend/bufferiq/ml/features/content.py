"""Content feature extraction."""

import re
from typing import Any, Dict, List

import pandas as pd

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.base import BaseFeatureExtractor

logger = get_logger(__name__)

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+")
HASHTAG_PATTERN = re.compile(r"#\w+")
MENTION_PATTERN = re.compile(r"@\w+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class ContentFeatureExtractor(BaseFeatureExtractor):
    """Extract content-based features from post text."""

    @property
    def feature_names(self) -> List[str]:
        """Return list of content feature names."""
        return [
            "text_length",
            "word_count",
            "avg_word_length",
            "sentence_count",
            "avg_sentence_length",
            "paragraph_count",
            "has_url",
            "url_count",
            "has_hashtag",
            "hashtag_count",
            "has_mention",
            "mention_count",
            "has_emoji",
            "emoji_count",
            "has_number",
            "number_count",
            "has_question",
            "question_count",
            "has_exclamation",
            "exclamation_count",
            "uppercase_ratio",
            "punctuation_ratio",
            "special_char_count",
            "newline_count",
            "whitespace_ratio",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract content features from DataFrame.

        Args:
            df: DataFrame with 'content' column

        Returns:
            DataFrame with content features
        """
        self.validate_input(df, ["content"])

        result = pd.DataFrame(index=df.index)

        # Fill missing content with empty string
        content_series = df["content"].fillna("")

        # Text length features
        result["text_length"] = content_series.str.len()

        # Word count (split on whitespace)
        words_series = content_series.str.split()
        result["word_count"] = words_series.str.len().fillna(0).astype(int)

        # Average word length
        result["avg_word_length"] = content_series.apply(
            lambda x: sum(len(word) for word in x.split()) / len(x.split())
            if x and len(x.split()) > 0
            else 0
        )

        # Sentence count (approximate: split on . ! ?)
        result["sentence_count"] = content_series.str.count(r"[.!?]+") + 1

        # Average sentence length
        result["avg_sentence_length"] = (
            result["word_count"] / result["sentence_count"]
        )

        # Paragraph count (double newlines)
        result["paragraph_count"] = content_series.str.count(r"\n\n") + 1

        # URL features
        result["has_url"] = content_series.str.contains(URL_PATTERN, regex=True).astype(int)
        result["url_count"] = content_series.apply(lambda x: len(URL_PATTERN.findall(x)))

        # Hashtag features
        result["has_hashtag"] = content_series.str.contains(HASHTAG_PATTERN, regex=True).astype(int)
        result["hashtag_count"] = content_series.apply(
            lambda x: len(HASHTAG_PATTERN.findall(x))
        )

        # Mention features
        result["has_mention"] = content_series.str.contains(MENTION_PATTERN, regex=True).astype(int)
        result["mention_count"] = content_series.apply(
            lambda x: len(MENTION_PATTERN.findall(x))
        )

        # Emoji features
        result["has_emoji"] = content_series.str.contains(EMOJI_PATTERN, regex=True).astype(int)
        result["emoji_count"] = content_series.apply(
            lambda x: len(EMOJI_PATTERN.findall(x))
        )

        # Number features
        result["has_number"] = content_series.str.contains(r"\d", regex=True).astype(int)
        result["number_count"] = content_series.str.count(r"\d+")

        # Question/exclamation features
        result["has_question"] = content_series.str.contains(r"\?").astype(int)
        result["question_count"] = content_series.str.count(r"\?")
        result["has_exclamation"] = content_series.str.contains(r"!").astype(int)
        result["exclamation_count"] = content_series.str.count(r"!")

        # Punctuation features
        result["uppercase_ratio"] = content_series.apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )

        result["punctuation_ratio"] = content_series.apply(
            lambda x: sum(1 for c in x if c in ".,;:!?-()[]{}\"'") / len(x)
            if len(x) > 0
            else 0
        )

        result["special_char_count"] = content_series.str.count(r"[!@#$%^&*()_+=]")

        # Newline features
        result["newline_count"] = content_series.str.count(r"\n")

        # Whitespace ratio
        result["whitespace_ratio"] = content_series.apply(
            lambda x: sum(1 for c in x if c.isspace()) / len(x) if len(x) > 0 else 0
        )

        logger.info(f"Extracted {len(result.columns)} content features")

        return result

    def extract_single(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content features from single post.

        Args:
            post_data: Dictionary with 'content' key

        Returns:
            Dictionary with content features
        """
        content = post_data.get("content", "")

        if not content:
            # Return zero features for empty content
            return {name: 0 for name in self.feature_names}

        # Text length
        text_length = len(content)
        words = content.split()
        word_count = len(words)

        # Average word length
        avg_word_length = (
            sum(len(word) for word in words) / word_count if word_count > 0 else 0
        )

        # Sentence count
        sentence_count = len(re.findall(r"[.!?]+", content)) + 1

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Paragraph count
        paragraph_count = content.count("\n\n") + 1

        # URLs
        urls = URL_PATTERN.findall(content)
        has_url = int(len(urls) > 0)
        url_count = len(urls)

        # Hashtags
        hashtags = HASHTAG_PATTERN.findall(content)
        has_hashtag = int(len(hashtags) > 0)
        hashtag_count = len(hashtags)

        # Mentions
        mentions = MENTION_PATTERN.findall(content)
        has_mention = int(len(mentions) > 0)
        mention_count = len(mentions)

        # Emojis
        emojis = EMOJI_PATTERN.findall(content)
        has_emoji = int(len(emojis) > 0)
        emoji_count = len(emojis)

        # Numbers
        has_number = int(bool(re.search(r"\d", content)))
        number_count = len(re.findall(r"\d+", content))

        # Questions/exclamations
        has_question = int("?" in content)
        question_count = content.count("?")
        has_exclamation = int("!" in content)
        exclamation_count = content.count("!")

        # Ratios
        uppercase_ratio = (
            sum(1 for c in content if c.isupper()) / text_length
            if text_length > 0
            else 0
        )

        punctuation_ratio = (
            sum(1 for c in content if c in ".,;:!?-()[]{}\"'") / text_length
            if text_length > 0
            else 0
        )

        special_char_count = len(re.findall(r"[!@#$%^&*()_+=]", content))
        newline_count = content.count("\n")

        whitespace_ratio = (
            sum(1 for c in content if c.isspace()) / text_length
            if text_length > 0
            else 0
        )

        features = {
            "text_length": text_length,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "paragraph_count": paragraph_count,
            "has_url": has_url,
            "url_count": url_count,
            "has_hashtag": has_hashtag,
            "hashtag_count": hashtag_count,
            "has_mention": has_mention,
            "mention_count": mention_count,
            "has_emoji": has_emoji,
            "emoji_count": emoji_count,
            "has_number": has_number,
            "number_count": number_count,
            "has_question": has_question,
            "question_count": question_count,
            "has_exclamation": has_exclamation,
            "exclamation_count": exclamation_count,
            "uppercase_ratio": uppercase_ratio,
            "punctuation_ratio": punctuation_ratio,
            "special_char_count": special_char_count,
            "newline_count": newline_count,
            "whitespace_ratio": whitespace_ratio,
        }

        return features