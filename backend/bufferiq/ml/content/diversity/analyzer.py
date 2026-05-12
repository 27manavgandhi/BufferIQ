"""
Content diversity analyzer.

Comprehensive diversity analysis across multiple dimensions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

from bufferiq.ml.content.diversity.topic_diversity import TopicDiversityAnalyzer
from bufferiq.ml.content.diversity.temporal_diversity import (
    TemporalDiversityAnalyzer,
)


@dataclass
class DiversityMetrics:
    """Content diversity measurements."""

    topic_diversity: float  # 0-1, Shannon entropy
    temporal_diversity: float  # 0-1
    platform_diversity: float  # 0-1
    sentiment_diversity: float  # 0-1
    repetition_score: float  # 0-1, lower = more repetitive
    overall_diversity: float  # 0-1


class ContentDiversityAnalyzer:
    """
        Analyze content diversity across posts.

        Measures:
        - Topic diversity (Shannon entropy)
        - Temporal diversity (posting patterns)
        - Platform diversity
        - Sentiment diversity
        - Content repetition

        Example:
    ```python
            analyzer = ContentDiversityAnalyzer()
            posts = [
                {"text": "AI post", "platform": "linkedin", "created_at": "2024-01-01"},
                {"text": "ML post", "platform": "twitter", "created_at": "2024-01-02"},
            ]
            metrics = analyzer.analyze(posts, window_days=30)
            print(f"Diversity: {metrics.overall_diversity:.2f}")
            if metrics.repetition_score > 0.7:
                print("Warning: High content repetition detected")
    ```
    """

    def __init__(self) -> None:
        """Initialize content diversity analyzer."""
        self.topic_analyzer = TopicDiversityAnalyzer()
        self.temporal_analyzer = TemporalDiversityAnalyzer()

    def analyze(self, posts: List[Dict], window_days: int = 30) -> DiversityMetrics:
        """
        Analyze content diversity.

        Args:
            posts: List of posts to analyze
            window_days: Time window for analysis

        Returns:
            Diversity metrics

        Raises:
            ValueError: If posts list is empty
        """
        if not posts:
            raise ValueError("Posts list cannot be empty")

        # Extract data from posts
        topics = self._extract_topics(posts)
        timestamps = self._extract_timestamps(posts)
        platforms = self._extract_platforms(posts)
        sentiments = self._extract_sentiments(posts)
        texts = [p.get("text", "") for p in posts]

        # Calculate individual metrics
        topic_diversity = (
            self.topic_analyzer.calculate_diversity(topics) if topics else 0.0
        )
        temporal_diversity = (
            self.temporal_analyzer.calculate_diversity(timestamps)
            if timestamps
            else 0.0
        )
        platform_diversity = self._calculate_platform_diversity(platforms)
        sentiment_diversity = self._calculate_sentiment_diversity(sentiments)
        repetition_score = self._calculate_repetition_score(texts)

        # Calculate overall diversity
        overall_diversity = np.mean(
            [
                topic_diversity,
                temporal_diversity,
                platform_diversity,
                sentiment_diversity,
                1.0 - repetition_score,  # Invert repetition
            ]
        )

        return DiversityMetrics(
            topic_diversity=topic_diversity,
            temporal_diversity=temporal_diversity,
            platform_diversity=platform_diversity,
            sentiment_diversity=sentiment_diversity,
            repetition_score=repetition_score,
            overall_diversity=overall_diversity,
        )

    def _extract_topics(self, posts: List[Dict]) -> List[str]:
        """Extract topics from posts."""
        topics = []
        for post in posts:
            # Use hashtags as proxy for topics
            text = post.get("text", "")
            hashtags = [word[1:] for word in text.split() if word.startswith("#")]
            topics.extend(hashtags)
        return topics if topics else ["general"]

    def _extract_timestamps(self, posts: List[Dict]) -> List[datetime]:
        """Extract timestamps from posts."""
        timestamps = []
        for post in posts:
            created_at = post.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    try:
                        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        timestamps.append(ts)
                    except Exception:
                        pass
                elif isinstance(created_at, datetime):
                    timestamps.append(created_at)
        return timestamps

    def _extract_platforms(self, posts: List[Dict]) -> List[str]:
        """Extract platforms from posts."""
        return [p.get("platform", "unknown") for p in posts]

    def _extract_sentiments(self, posts: List[Dict]) -> List[str]:
        """Extract sentiments from posts."""
        # Simplified - would integrate with sentiment analyzer
        sentiments = []
        for post in posts:
            sentiment = post.get("sentiment", "neutral")
            sentiments.append(sentiment)
        return sentiments

    def _calculate_platform_diversity(self, platforms: List[str]) -> float:
        """Calculate platform diversity."""
        if not platforms:
            return 0.0
        return self.topic_analyzer.calculate_diversity(platforms)

    def _calculate_sentiment_diversity(self, sentiments: List[str]) -> float:
        """Calculate sentiment diversity."""
        if not sentiments:
            return 0.0
        return self.topic_analyzer.calculate_diversity(sentiments)

    def _calculate_repetition_score(self, texts: List[str]) -> float:
        """
        Calculate content repetition score.

        Higher score = more repetition.

        Args:
            texts: List of text content

        Returns:
            Repetition score (0-1)
        """
        if len(texts) < 2:
            return 0.0

        # Calculate pairwise similarity (simplified)
        total_pairs = 0
        similar_pairs = 0

        for i, text1 in enumerate(texts):
            for text2 in texts[i + 1 :]:
                total_pairs += 1
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity > 0.7:  # Threshold for "similar"
                    similar_pairs += 1

        repetition = similar_pairs / total_pairs if total_pairs > 0 else 0.0
        return repetition

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity (simplified Jaccard).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0
