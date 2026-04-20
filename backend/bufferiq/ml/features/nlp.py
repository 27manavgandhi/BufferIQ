"""NLP feature extraction."""

from typing import Any

import pandas as pd

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.base import BaseFeatureExtractor

logger = get_logger(__name__)

# Try importing TextBlob, textstat
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available, sentiment features will be disabled")

try:
    import textstat

    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logger.warning("textstat not available, readability features will be disabled")


class NLPFeatureExtractor(BaseFeatureExtractor):
    """Extract NLP features from post text."""

    @property
    def feature_names(self) -> list[str]:
        """Return list of NLP feature names."""
        return [
            "sentiment_polarity",
            "sentiment_subjectivity",
            "sentiment_label",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "automated_readability_index",
            "coleman_liau_index",
            "avg_readability",
            "lexical_diversity",
            "stopword_ratio",
            "noun_count",
            "verb_count",
            "adjective_count",
            "adverb_count",
            "proper_noun_count",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract NLP features from DataFrame.

        Args:
            df: DataFrame with 'content' column

        Returns:
            DataFrame with NLP features
        """
        self.validate_input(df, ["content"])

        result = pd.DataFrame(index=df.index)

        # Fill missing content
        content_series = df["content"].fillna("")

        # Sentiment features
        if TEXTBLOB_AVAILABLE:
            sentiments = content_series.apply(self._extract_sentiment)
            result["sentiment_polarity"] = sentiments.apply(lambda x: x[0])
            result["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])
            result["sentiment_label"] = sentiments.apply(lambda x: x[2])
        else:
            result["sentiment_polarity"] = 0.0
            result["sentiment_subjectivity"] = 0.0
            result["sentiment_label"] = 0  # neutral

        # Readability features
        if TEXTSTAT_AVAILABLE:
            result["flesch_reading_ease"] = content_series.apply(
                self._safe_flesch_reading_ease
            )
            result["flesch_kincaid_grade"] = content_series.apply(
                self._safe_flesch_kincaid_grade
            )
            result["automated_readability_index"] = content_series.apply(
                self._safe_automated_readability_index
            )
            result["coleman_liau_index"] = content_series.apply(
                self._safe_coleman_liau_index
            )

            # Average readability (normalized)
            result["avg_readability"] = (
                result["flesch_reading_ease"] / 100
                + (100 - result["flesch_kincaid_grade"]) / 100
                + (100 - result["automated_readability_index"]) / 100
                + (100 - result["coleman_liau_index"]) / 100
            ) / 4
        else:
            result["flesch_reading_ease"] = 0.0
            result["flesch_kincaid_grade"] = 0.0
            result["automated_readability_index"] = 0.0
            result["coleman_liau_index"] = 0.0
            result["avg_readability"] = 0.0

        # Linguistic features
        result["lexical_diversity"] = content_series.apply(self._lexical_diversity)
        result["stopword_ratio"] = content_series.apply(self._stopword_ratio)

        # POS features (simple regex-based, not as accurate as spaCy but faster)
        result["noun_count"] = content_series.apply(self._count_nouns)
        result["verb_count"] = content_series.apply(self._count_verbs)
        result["adjective_count"] = content_series.apply(self._count_adjectives)
        result["adverb_count"] = content_series.apply(self._count_adverbs)
        result["proper_noun_count"] = content_series.apply(self._count_proper_nouns)

        logger.info(f"Extracted {len(result.columns)} NLP features")

        return result

    def extract_single(self, post_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract NLP features from single post.

        Args:
            post_data: Dictionary with 'content' key

        Returns:
            Dictionary with NLP features
        """
        content = post_data.get("content", "")

        if not content:
            return {name: 0 for name in self.feature_names}

        # Sentiment
        (
            sentiment_polarity,
            sentiment_subjectivity,
            sentiment_label,
        ) = self._extract_sentiment(content)

        # Readability
        flesch_reading_ease = self._safe_flesch_reading_ease(content)
        flesch_kincaid_grade = self._safe_flesch_kincaid_grade(content)
        automated_readability_index = self._safe_automated_readability_index(content)
        coleman_liau_index = self._safe_coleman_liau_index(content)

        avg_readability = (
            flesch_reading_ease / 100
            + (100 - flesch_kincaid_grade) / 100
            + (100 - automated_readability_index) / 100
            + (100 - coleman_liau_index) / 100
        ) / 4

        # Linguistic features
        lexical_diversity = self._lexical_diversity(content)
        stopword_ratio = self._stopword_ratio(content)
        noun_count = self._count_nouns(content)
        verb_count = self._count_verbs(content)
        adjective_count = self._count_adjectives(content)
        adverb_count = self._count_adverbs(content)
        proper_noun_count = self._count_proper_nouns(content)

        features = {
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "sentiment_label": sentiment_label,
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "automated_readability_index": automated_readability_index,
            "coleman_liau_index": coleman_liau_index,
            "avg_readability": avg_readability,
            "lexical_diversity": lexical_diversity,
            "stopword_ratio": stopword_ratio,
            "noun_count": noun_count,
            "verb_count": verb_count,
            "adjective_count": adjective_count,
            "adverb_count": adverb_count,
            "proper_noun_count": proper_noun_count,
        }

        return features

    def _extract_sentiment(self, text: str) -> tuple[float, float, int]:
        """Extract sentiment polarity, subjectivity, and label."""
        if not text or not TEXTBLOB_AVAILABLE:
            return 0.0, 0.0, 0

        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Label: -1 (negative), 0 (neutral), 1 (positive)
            if polarity < -0.1:
                label = -1
            elif polarity > 0.1:
                label = 1
            else:
                label = 0

            return float(polarity), float(subjectivity), int(label)
        except Exception as e:
            logger.warning(f"Sentiment extraction failed: {e}")
            return 0.0, 0.0, 0

    def _safe_flesch_reading_ease(self, text: str) -> float:
        """Safely calculate Flesch Reading Ease."""
        if not text or not TEXTSTAT_AVAILABLE:
            return 0.0

        try:
            score = textstat.flesch_reading_ease(text)
            return max(0.0, min(100.0, float(score)))
        except Exception:
            return 0.0

    def _safe_flesch_kincaid_grade(self, text: str) -> float:
        """Safely calculate Flesch-Kincaid Grade."""
        if not text or not TEXTSTAT_AVAILABLE:
            return 0.0

        try:
            score = textstat.flesch_kincaid_grade(text)
            return max(0.0, float(score))
        except Exception:
            return 0.0

    def _safe_automated_readability_index(self, text: str) -> float:
        """Safely calculate Automated Readability Index."""
        if not text or not TEXTSTAT_AVAILABLE:
            return 0.0

        try:
            score = textstat.automated_readability_index(text)
            return max(0.0, float(score))
        except Exception:
            return 0.0

    def _safe_coleman_liau_index(self, text: str) -> float:
        """Safely calculate Coleman-Liau Index."""
        if not text or not TEXTSTAT_AVAILABLE:
            return 0.0

        try:
            score = textstat.coleman_liau_index(text)
            return max(0.0, float(score))
        except Exception:
            return 0.0

    def _lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (unique words / total words)."""
        if not text:
            return 0.0

        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        unique_words = len(set(words))
        return unique_words / len(words)

    def _stopword_ratio(self, text: str) -> float:
        """Calculate proportion of stopwords."""
        # Common English stopwords
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
        }

        if not text:
            return 0.0

        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        stopword_count = sum(1 for word in words if word in stopwords)
        return stopword_count / len(words)

    def _count_nouns(self, text: str) -> int:
        """Count nouns (simple heuristic: words ending in common noun suffixes)."""
        if not text:
            return 0

        words = text.lower().split()
        noun_suffixes = ["tion", "ment", "ness", "ity", "ism", "ship", "ance", "ence"]
        count = sum(1 for word in words if any(word.endswith(s) for s in noun_suffixes))
        return count

    def _count_verbs(self, text: str) -> int:
        """Count verbs (simple heuristic: words ending in common verb suffixes)."""
        if not text:
            return 0

        words = text.lower().split()
        verb_suffixes = ["ed", "ing", "ize", "ise", "ate", "ify"]
        count = sum(1 for word in words if any(word.endswith(s) for s in verb_suffixes))
        return count

    def _count_adjectives(self, text: str) -> int:
        """Count adjectives (simple heuristic: words ending in common adj suffixes)."""
        if not text:
            return 0

        words = text.lower().split()
        adj_suffixes = ["able", "ible", "ful", "less", "ous", "ive", "al"]
        count = sum(1 for word in words if any(word.endswith(s) for s in adj_suffixes))
        return count

    def _count_adverbs(self, text: str) -> int:
        """Count adverbs (simple heuristic: words ending in 'ly')."""
        if not text:
            return 0

        words = text.lower().split()
        count = sum(1 for word in words if word.endswith("ly"))
        return count

    def _count_proper_nouns(self, text: str) -> int:
        """Count proper nouns (simple heuristic: capitalized words not at start)."""
        if not text:
            return 0

        # Split into words, skip first word of sentences
        words = text.split()
        count = sum(
            1
            for i, word in enumerate(words)
            if i > 0
            and word
            and word[0].isupper()
            and not word.isupper()
            and (i == 0 or words[i - 1][-1] not in ".!?")
        )
        return count
