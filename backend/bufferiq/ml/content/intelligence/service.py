"""
Content intelligence service.

Main orchestrator for all content analysis modules.
"""

from typing import Any, Dict, List, Optional

from bufferiq.ml.content.preprocessing.text_cleaner import TextCleaner
from bufferiq.ml.content.preprocessing.feature_extractor import (
    TextFeatureExtractor,
)
from bufferiq.ml.content.sentiment.analyzer import SentimentAnalyzer
from bufferiq.ml.content.readability.analyzer import ReadabilityAnalyzer
from bufferiq.ml.content.quality.content_validator import (
    ContentQualityChecker,
)
from bufferiq.ml.content.optimizer.optimizer import ContentOptimizer

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ContentIntelligenceService:
    """
        Main orchestrator for content intelligence.

        Coordinates all content analysis modules:
        - Text preprocessing
        - Sentiment analysis
        - Topic modeling
        - Readability analysis
        - Quality checking
        - Optimization
        - Diversity analysis

        Example:
    ```python
            service = ContentIntelligenceService()

            # Analyze single post
            analysis = service.analyze_content(
                text="Great post about AI!",
                platform="linkedin"
            )

            # Batch analysis
            results = service.analyze_batch(posts)
    ```
    """

    def __init__(
        self,
        text_cleaner: Optional[TextCleaner] = None,
        feature_extractor: Optional[TextFeatureExtractor] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        readability_analyzer: Optional[ReadabilityAnalyzer] = None,
        quality_checker: Optional[ContentQualityChecker] = None,
        optimizer: Optional[ContentOptimizer] = None,
    ) -> None:
        """
        Initialize content intelligence service.

        Args:
            text_cleaner: Optional text cleaner instance
            feature_extractor: Optional feature extractor
            sentiment_analyzer: Optional sentiment analyzer
            readability_analyzer: Optional readability analyzer
            quality_checker: Optional quality checker
            optimizer: Optional optimizer
        """
        self.text_cleaner = text_cleaner or TextCleaner()
        self.feature_extractor = feature_extractor or TextFeatureExtractor()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.readability_analyzer = readability_analyzer or ReadabilityAnalyzer()
        self.quality_checker = quality_checker or ContentQualityChecker()
        self.optimizer = optimizer or ContentOptimizer()

    def analyze_content(
        self,
        text: str,
        platform: str,
        user_id: Optional[str] = None,
        include_optimization: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive content analysis.

        Args:
            text: Content to analyze
            platform: Platform type
            user_id: Optional user ID for personalization
            include_optimization: Include optimization suggestions

        Returns:
            Complete analysis results

        Raises:
            ValueError: If platform not supported or text invalid
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Preprocess text
        preprocessed = self.text_cleaner.clean(text)

        # Extract features
        features = self.feature_extractor.extract(preprocessed)

        # Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze(text)

        # Readability analysis
        try:
            readability = self.readability_analyzer.analyze(text)
        except ValueError:
            # Text too short for readability
            readability = None

        # Quality check
        quality = self.quality_checker.check(text, platform)

        # Build analysis result
        analysis = {
            "text": text,
            "platform": platform,
            "preprocessed": {
                "cleaned": preprocessed.cleaned,
                "language": preprocessed.language,
                "word_count": preprocessed.word_count,
                "char_count": preprocessed.char_count,
                "sentence_count": preprocessed.sentence_count,
                "hashtags": preprocessed.hashtags,
                "mentions": preprocessed.mentions,
                "urls": preprocessed.urls,
                "emojis": preprocessed.emojis,
            },
            "features": {
                "word_count": features.word_count,
                "char_count": features.char_count,
                "sentence_count": features.sentence_count,
                "avg_word_length": features.avg_word_length,
                "avg_sentence_length": features.avg_sentence_length,
                "has_url": features.has_url,
                "url_count": features.url_count,
                "has_hashtag": features.has_hashtag,
                "hashtag_count": features.hashtag_count,
                "has_mention": features.has_mention,
                "mention_count": features.mention_count,
                "has_emoji": features.has_emoji,
                "emoji_count": features.emoji_count,
                "has_question": features.has_question,
                "has_exclamation": features.has_exclamation,
            },
            "sentiment": {
                "sentiment": sentiment.sentiment.value,
                "confidence": sentiment.confidence,
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
                "scores": sentiment.scores,
            },
            "quality": {
                "score": quality.score,
                "grammar_errors": quality.grammar_errors,
                "spelling_errors": quality.spelling_errors,
                "broken_links": quality.broken_links,
                "warnings": quality.warnings,
                "recommendations": quality.recommendations,
            },
        }

        # Add readability if available
        if readability:
            analysis["readability"] = {
                "flesch_reading_ease": readability.flesch_reading_ease,
                "flesch_kincaid_grade": readability.flesch_kincaid_grade,
                "gunning_fog": readability.gunning_fog,
                "average_grade_level": readability.average_grade_level,
                "reading_difficulty": readability.reading_difficulty,
            }

        # Add optimization if requested
        if include_optimization:
            optimization = self.optimizer.optimize(text, platform, analysis=analysis)
            analysis["optimization"] = {
                "overall_score": optimization.overall_score,
                "predicted_engagement_lift": optimization.predicted_engagement_lift,
                "best_platform": optimization.best_platform,
                "suggestions": [
                    {
                        "type": s.type,
                        "priority": s.priority,
                        "current_value": s.current_value,
                        "suggested_value": s.suggested_value,
                        "impact": s.impact,
                        "confidence": s.confidence,
                    }
                    for s in optimization.suggestions
                ],
                "rewrite_examples": optimization.rewrite_examples,
            }

        return analysis

    def analyze_batch(
        self, posts: List[Dict[str, Any]], platform: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple posts.

        Args:
            posts: List of posts with 'text' field
            platform: Platform type

        Returns:
            List of analysis results

        Raises:
            ValueError: If posts list is empty or platform invalid
        """
        if not posts:
            raise ValueError("Posts list cannot be empty")

        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        results = []
        for post in posts:
            text = post.get("text", "")
            if text and text.strip():
                try:
                    analysis = self.analyze_content(text, platform)
                    results.append(analysis)
                except Exception as e:
                    # Log error and continue
                    results.append(
                        {"text": text, "error": str(e), "platform": platform}
                    )

        return results
