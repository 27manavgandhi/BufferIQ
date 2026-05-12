"""
Sentiment analysis using TextBlob and VADER.

Provides multi-class sentiment detection with confidence scores.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Sentiment(str, Enum):
    """Sentiment categories."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""

    sentiment: Sentiment
    confidence: float
    scores: Dict[str, float]
    subjectivity: float
    polarity: float


class SentimentAnalyzer:
    """
        Analyze sentiment in text.

        Uses both TextBlob and VADER for robust sentiment detection.
        Combines results for higher accuracy.

        Example:
    ```python
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("I love this product!")
            print(result.sentiment)  # Sentiment.POSITIVE
            print(result.confidence)  # 0.95
    ```
    """

    def __init__(self) -> None:
        """Initialize sentiment analyzer."""
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment in text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis result

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores["compound"]

        # TextBlob analysis
        blob = TextBlob(text)
        blob_polarity = blob.sentiment.polarity
        blob_subjectivity = blob.sentiment.subjectivity

        # Combine scores (weighted average)
        combined_polarity = (vader_compound * 0.6) + (blob_polarity * 0.4)

        # Determine sentiment
        if combined_polarity >= 0.05:
            sentiment = Sentiment.POSITIVE
            confidence = min(abs(combined_polarity), 1.0)
        elif combined_polarity <= -0.05:
            sentiment = Sentiment.NEGATIVE
            confidence = min(abs(combined_polarity), 1.0)
        else:
            sentiment = Sentiment.NEUTRAL
            confidence = 1.0 - abs(combined_polarity)

        scores = {
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"],
            "compound": vader_compound,
        }

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            subjectivity=blob_subjectivity,
            polarity=combined_polarity,
        )
