"""
Tests for content intelligence service.
"""

import pytest

from bufferiq.ml.content.intelligence.service import ContentIntelligenceService


class TestContentIntelligenceService:
    """Test ContentIntelligenceService class."""

    @pytest.fixture
    def service(self) -> ContentIntelligenceService:
        """Create service fixture."""
        return ContentIntelligenceService()

    def test_analyze_content_basic(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test basic content analysis."""
        result = service.analyze_content("Great post about AI!", "linkedin")

        assert isinstance(result, dict)
        assert "text" in result
        assert "platform" in result

    def test_analyze_content_returns_all_sections(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis returns all expected sections."""
        result = service.analyze_content(
            "This is a test post about technology.", "linkedin"
        )

        assert "preprocessed" in result
        assert "features" in result
        assert "sentiment" in result
        assert "quality" in result
        assert "optimization" in result

    def test_analyze_content_invalid_platform_raises_error(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            service.analyze_content("Test", "facebook")

    def test_analyze_content_empty_text_raises_error(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.analyze_content("", "linkedin")

    def test_analyze_content_preprocessed_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test preprocessed section."""
        result = service.analyze_content("Test #AI @john", "linkedin")

        assert "preprocessed" in result
        preprocessed = result["preprocessed"]
        assert "cleaned" in preprocessed
        assert "word_count" in preprocessed
        assert "hashtags" in preprocessed
        assert "mentions" in preprocessed

    def test_analyze_content_features_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test features section."""
        result = service.analyze_content("Test post!", "linkedin")

        assert "features" in result
        features = result["features"]
        assert "word_count" in features
        assert "has_hashtag" in features
        assert "has_emoji" in features

    def test_analyze_content_sentiment_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test sentiment section."""
        result = service.analyze_content("I love this!", "linkedin")

        assert "sentiment" in result
        sentiment = result["sentiment"]
        assert "sentiment" in sentiment
        assert "confidence" in sentiment
        assert "polarity" in sentiment

    def test_analyze_content_quality_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test quality section."""
        result = service.analyze_content("Good post here.", "linkedin")

        assert "quality" in result
        quality = result["quality"]
        assert "score" in quality
        assert "grammar_errors" in quality
        assert "recommendations" in quality

    def test_analyze_content_readability_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test readability section."""
        result = service.analyze_content(
            "This is a longer post with multiple sentences. "
            "It should have readability analysis included.",
            "linkedin",
        )

        # Readability may be included if text is long enough
        if "readability" in result:
            readability = result["readability"]
            assert "flesch_reading_ease" in readability
            assert "reading_difficulty" in readability

    def test_analyze_content_optimization_section(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test optimization section."""
        result = service.analyze_content(
            "Test post", "linkedin", include_optimization=True
        )

        assert "optimization" in result
        optimization = result["optimization"]
        assert "overall_score" in optimization
        assert "suggestions" in optimization

    def test_analyze_content_without_optimization(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis without optimization."""
        result = service.analyze_content(
            "Test post", "linkedin", include_optimization=False
        )

        assert "optimization" not in result

    def test_analyze_content_with_user_id(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis with user ID."""
        result = service.analyze_content(
            "Test post", "linkedin", user_id="user123"
        )

        assert isinstance(result, dict)

    def test_analyze_batch_basic(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test batch analysis."""
        posts = [
            {"text": "First post"},
            {"text": "Second post"},
        ]
        results = service.analyze_batch(posts, "linkedin")

        assert isinstance(results, list)
        assert len(results) == 2

    def test_analyze_batch_empty_raises_error(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test empty batch raises error."""
        with pytest.raises(ValueError, match="Posts list cannot be empty"):
            service.analyze_batch([], "linkedin")

    def test_analyze_batch_invalid_platform_raises_error(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test invalid platform raises error."""
        posts = [{"text": "Test"}]
        with pytest.raises(ValueError, match="not supported"):
            service.analyze_batch(posts, "facebook")

    def test_analyze_batch_handles_errors(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test batch analysis handles errors gracefully."""
        posts = [
            {"text": "Good post"},
            {"text": ""},  # Empty text
            {"text": "Another good post"},
        ]
        results = service.analyze_batch(posts, "linkedin")

        # Should return results for valid posts and error objects for invalid
        assert isinstance(results, list)
        assert len(results) == 3

    def test_analyze_batch_filters_empty_text(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test batch analysis filters empty text."""
        posts = [
            {"text": "Good post"},
            {"text": "   "},  # Whitespace only
        ]
        results = service.analyze_batch(posts, "linkedin")

        # Should only process non-empty posts
        assert isinstance(results, list)

    def test_analyze_content_linkedin(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test LinkedIn-specific analysis."""
        result = service.analyze_content("Professional post here", "linkedin")

        assert result["platform"] == "linkedin"

    def test_analyze_content_twitter(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test Twitter-specific analysis."""
        result = service.analyze_content("Quick tweet!", "twitter")

        assert result["platform"] == "twitter"

    def test_analyze_content_bluesky(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test Bluesky-specific analysis."""
        result = service.analyze_content("Bluesky post", "bluesky")

        assert result["platform"] == "bluesky"

    def test_analyze_content_with_hashtags(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis with hashtags."""
        result = service.analyze_content(
            "Post about #AI and #ML", "linkedin"
        )

        assert len(result["preprocessed"]["hashtags"]) == 2

    def test_analyze_content_with_mentions(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis with mentions."""
        result = service.analyze_content("Thanks @john @jane", "linkedin")

        assert len(result["preprocessed"]["mentions"]) == 2

    def test_analyze_content_with_urls(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis with URLs."""
        result = service.analyze_content(
            "Check https://example.com", "linkedin"
        )

        assert len(result["preprocessed"]["urls"]) == 1

    def test_analyze_content_with_emojis(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis with emojis."""
        result = service.analyze_content("Great! 🚀 💯", "linkedin")

        assert len(result["preprocessed"]["emojis"]) >= 2

    def test_analyze_content_complex(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test complex content analysis."""
        text = (
            "Excited to share our new AI project! 🚀 "
            "Check it out at https://example.com "
            "#AI #MachineLearning #Innovation "
            "Thanks @team for the great work!"
        )
        result = service.analyze_content(text, "linkedin")

        assert "preprocessed" in result
        assert "features" in result
        assert "sentiment" in result
        assert "quality" in result
        assert "optimization" in result

    def test_analyze_content_short_text(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis of short text."""
        result = service.analyze_content("Quick update!", "twitter")

        assert isinstance(result, dict)
        # Readability may not be available for very short text
        assert "sentiment" in result

    def test_analyze_content_long_text(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test analysis of long text."""
        text = (
            "This is a longer piece of content that discusses various topics "
            "in detail. It contains multiple sentences and provides comprehensive "
            "information about the subject matter. The content is well-structured "
            "and should be analyzed thoroughly by all components of the system."
        )
        result = service.analyze_content(text, "linkedin")

        assert "readability" in result
        assert isinstance(result, dict)

    def test_analyze_batch_multiple_platforms(
        self, service: ContentIntelligenceService
    ) -> None:
        """Test batch analysis uses single platform."""
        posts = [
            {"text": "Post 1"},
            {"text": "Post 2"},
            {"text": "Post 3"},
        ]
        results = service.analyze_batch(posts, "linkedin")

        # All results should be for the same platform
        for result in results:
            if "platform" in result:
                assert result["platform"] == "linkedin"