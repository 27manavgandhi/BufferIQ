"""
Tests for content validator.
"""

import pytest

from bufferiq.ml.content.quality.content_validator import (
    ContentQualityChecker,
    QualityReport,
)


class TestContentQualityChecker:
    """Test ContentQualityChecker class."""

    @pytest.fixture
    def checker(self) -> ContentQualityChecker:
        """Create quality checker fixture."""
        return ContentQualityChecker()

    def test_check_basic(self, checker: ContentQualityChecker) -> None:
        """Test basic quality check."""
        report = checker.check("This is a good post.", "linkedin")

        assert isinstance(report, QualityReport)
        assert 0.0 <= report.score <= 100.0

    def test_check_with_errors(self, checker: ContentQualityChecker) -> None:
        """Test checking text with errors."""
        report = checker.check("This are wrong.", "linkedin")

        assert report.grammar_errors > 0
        assert report.score < 100

    def test_check_invalid_platform_raises_error(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            checker.check("Test", "facebook")

    def test_check_empty_text_raises_error(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            checker.check("", "linkedin")

    def test_check_returns_issues(self, checker: ContentQualityChecker) -> None:
        """Test that check returns issues."""
        report = checker.check("This are wrong.", "linkedin")

        assert isinstance(report.issues, list)

    def test_check_returns_recommendations(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test that check returns recommendations."""
        report = checker.check("Test post here.", "linkedin")

        assert isinstance(report.recommendations, list)

    def test_check_too_long_text(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test checking text that exceeds platform limit."""
        # Twitter has 280 character limit
        long_text = "x" * 300
        report = checker.check(long_text, "twitter")

        # Should have length error
        assert any(issue.type == "length" for issue in report.issues)

    def test_check_too_short_text(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test checking very short text."""
        report = checker.check("Hi", "linkedin")

        # May have warning about short text
        assert isinstance(report, QualityReport)

    def test_check_with_links(self, checker: ContentQualityChecker) -> None:
        """Test checking text with links."""
        report = checker.check(
            "Check https://example.com", "linkedin", include_links=True
        )

        assert isinstance(report, QualityReport)

    def test_check_without_links(self, checker: ContentQualityChecker) -> None:
        """Test checking without link validation."""
        report = checker.check(
            "Check https://example.com", "linkedin", include_links=False
        )

        assert report.broken_links == 0

    def test_check_score_calculation(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test score calculation."""
        report = checker.check("This is a perfect post.", "linkedin")

        # Perfect text should score high
        assert report.score > 80

    def test_check_linkedin_limits(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test LinkedIn-specific limits."""
        text = "x" * 100
        report = checker.check(text, "linkedin")

        # Should pass LinkedIn limits
        assert isinstance(report, QualityReport)

    def test_check_twitter_limits(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test Twitter-specific limits."""
        text = "Short tweet here!"
        report = checker.check(text, "twitter")

        assert isinstance(report, QualityReport)

    def test_check_bluesky_limits(
        self, checker: ContentQualityChecker
    ) -> None:
        """Test Bluesky-specific limits."""
        text = "Short post here!"
        report = checker.check(text, "bluesky")

        assert isinstance(report, QualityReport)