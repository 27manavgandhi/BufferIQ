"""
Tests for readability scorer.
"""

import pytest

from bufferiq.ml.content.readability.scorer import ReadabilityScorer


class TestReadabilityScorer:
    """Test ReadabilityScorer class."""

    @pytest.fixture
    def scorer(self) -> ReadabilityScorer:
        """Create readability scorer fixture."""
        return ReadabilityScorer()

    def test_score_linkedin(self, scorer: ReadabilityScorer) -> None:
        """Test scoring for LinkedIn."""
        text = "This is a professional post about business topics."
        result = scorer.score(text, "linkedin")

        assert "score" in result
        assert "scores" in result
        assert "recommendations" in result

    def test_score_twitter(self, scorer: ReadabilityScorer) -> None:
        """Test scoring for Twitter."""
        text = "Quick update about today's events!"
        result = scorer.score(text, "twitter")

        assert "score" in result
        assert 0.0 <= result["score"] <= 100.0

    def test_score_bluesky(self, scorer: ReadabilityScorer) -> None:
        """Test scoring for Bluesky."""
        text = "Sharing some thoughts here."
        result = scorer.score(text, "bluesky")

        assert "score" in result

    def test_score_invalid_platform_raises_error(
        self, scorer: ReadabilityScorer
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            scorer.score("Test", "facebook")

    def test_score_too_short_raises_error(
        self, scorer: ReadabilityScorer
    ) -> None:
        """Test too short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            scorer.score("Hi", "linkedin")

    def test_score_returns_recommendations(
        self, scorer: ReadabilityScorer
    ) -> None:
        """Test that scoring returns recommendations."""
        text = "This is a test post."
        result = scorer.score(text, "linkedin")

        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_score_platform_targets(self, scorer: ReadabilityScorer) -> None:
        """Test platform targets are returned."""
        text = "Test post here."
        result = scorer.score(text, "linkedin")

        assert "platform_targets" in result
        assert "max_grade" in result["platform_targets"]

    def test_score_high_quality_text(self, scorer: ReadabilityScorer) -> None:
        """Test scoring high quality text."""
        text = "This is a well-written and clear post about important topics."
        result = scorer.score(text, "linkedin")

        # High quality should score well
        assert result["score"] > 50

    def test_score_complex_text(self, scorer: ReadabilityScorer) -> None:
        """Test scoring complex text."""
        text = (
            "The implementation of sophisticated methodologies "
            "necessitates comprehensive understanding of complex principles."
        )
        result = scorer.score(text, "linkedin")

        # Complex text may score lower
        assert isinstance(result["score"], float)