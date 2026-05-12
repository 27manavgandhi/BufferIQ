"""
Tests for content scorer.
"""

import pytest

from bufferiq.ml.content.optimizer.scorer import ContentScorer


class TestContentScorer:
    """Test ContentScorer class."""

    @pytest.fixture
    def scorer(self) -> ContentScorer:
        """Create content scorer fixture."""
        return ContentScorer()

    def test_score_basic(self, scorer: ContentScorer) -> None:
        """Test basic scoring."""
        analysis = {
            "quality": {"score": 90.0},
            "readability": {"reading_difficulty": "easy"},
            "sentiment": {"confidence": 0.8},
            "features": {"hashtag_count": 3, "emoji_count": 1},
        }
        score = scorer.score("Test post here", "linkedin", analysis)

        assert 0.0 <= score <= 100.0

    def test_score_invalid_platform_raises_error(
        self, scorer: ContentScorer
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            scorer.score("Test", "facebook", {})

    def test_score_high_quality(self, scorer: ContentScorer) -> None:
        """Test scoring high quality content."""
        analysis = {
            "quality": {"score": 95.0},
            "readability": {"reading_difficulty": "easy"},
            "sentiment": {"confidence": 0.9},
            "features": {"hashtag_count": 3, "emoji_count": 2, "url_count": 1},
        }
        score = scorer.score("x" * 200, "linkedin", analysis)

        assert score > 70

    def test_score_low_quality(self, scorer: ContentScorer) -> None:
        """Test scoring low quality content."""
        analysis = {
            "quality": {"score": 30.0},
            "readability": {"reading_difficulty": "hard"},
            "sentiment": {"confidence": 0.3},
            "features": {"hashtag_count": 0},
        }
        score = scorer.score("Test", "linkedin", analysis)

        assert score < 70

    def test_score_optimal_length(self, scorer: ContentScorer) -> None:
        """Test scoring optimal length content."""
        analysis = {"features": {"hashtag_count": 3}}
        text = "x" * 200  # Optimal for LinkedIn
        score = scorer.score(text, "linkedin", analysis)

        # Should score well for length
        assert score > 50

    def test_score_too_short(self, scorer: ContentScorer) -> None:
        """Test scoring too short content."""
        analysis = {"features": {"hashtag_count": 0}}
        score = scorer.score("Hi", "linkedin", analysis)

        # Short content scores lower
        assert isinstance(score, float)

    def test_score_too_long(self, scorer: ContentScorer) -> None:
        """Test scoring too long content."""
        analysis = {"features": {"hashtag_count": 0}}
        text = "x" * 1000
        score = scorer.score(text, "linkedin", analysis)

        # Very long content may score lower
        assert isinstance(score, float)

    def test_score_with_hashtags(self, scorer: ContentScorer) -> None:
        """Test scoring with hashtags."""
        analysis = {"features": {"hashtag_count": 3, "emoji_count": 0}}
        score = scorer.score("Test post", "linkedin", analysis)

        # Hashtags improve score
        assert score > 50

    def test_score_with_emojis(self, scorer: ContentScorer) -> None:
        """Test scoring with emojis."""
        analysis = {"features": {"hashtag_count": 0, "emoji_count": 2}}
        score = scorer.score("Test post", "linkedin", analysis)

        # Emojis improve score
        assert score > 50

    def test_score_with_url(self, scorer: ContentScorer) -> None:
        """Test scoring with URL."""
        analysis = {
            "features": {"hashtag_count": 0, "emoji_count": 0, "url_count": 1}
        }
        score = scorer.score("Test post", "linkedin", analysis)

        # URL improves score
        assert score > 50

    def test_score_missing_features(self, scorer: ContentScorer) -> None:
        """Test scoring with missing features."""
        analysis = {}
        score = scorer.score("Test post", "linkedin", analysis)

        # Should handle missing features gracefully
        assert 0.0 <= score <= 100.0

    def test_score_twitter_length(self, scorer: ContentScorer) -> None:
        """Test Twitter-specific length scoring."""
        analysis = {"features": {"hashtag_count": 1}}
        text = "x" * 150  # Good for Twitter
        score = scorer.score(text, "twitter", analysis)

        assert isinstance(score, float)

    def test_score_bluesky_length(self, scorer: ContentScorer) -> None:
        """Test Bluesky-specific length scoring."""
        analysis = {"features": {"hashtag_count": 1}}
        text = "x" * 150
        score = scorer.score(text, "bluesky", analysis)

        assert isinstance(score, float)

    def test_score_rounded(self, scorer: ContentScorer) -> None:
        """Test score is properly rounded."""
        analysis = {"features": {"hashtag_count": 1}}
        score = scorer.score("Test", "linkedin", analysis)

        # Should be rounded to 1 decimal place
        assert score == round(score, 1)