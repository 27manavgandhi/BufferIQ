"""
Tests for link validator.
"""

import pytest

from bufferiq.ml.content.quality.link_validator import (
    LinkValidator,
    LinkValidation,
)


class TestLinkValidator:
    """Test LinkValidator class."""

    @pytest.fixture
    def validator(self) -> LinkValidator:
        """Create link validator fixture."""
        return LinkValidator()

    def test_validate_links_basic(self, validator: LinkValidator) -> None:
        """Test basic link validation."""
        results = validator.validate_links(
            "Check https://example.com for info"
        )

        assert len(results) == 1
        assert isinstance(results[0], LinkValidation)

    def test_validate_links_multiple(self, validator: LinkValidator) -> None:
        """Test validating multiple links."""
        text = "Visit https://example.com and http://test.com"
        results = validator.validate_links(text)

        assert len(results) == 2

    def test_validate_links_no_links(self, validator: LinkValidator) -> None:
        """Test text with no links."""
        results = validator.validate_links("No links here")

        assert len(results) == 0

    def test_validate_links_empty_text_raises_error(
        self, validator: LinkValidator
    ) -> None:
        """Test validating empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            validator.validate_links("")

    def test_validate_link_https(self, validator: LinkValidator) -> None:
        """Test validating HTTPS link."""
        result = validator.validate_link("https://example.com")

        assert result.is_valid is True
        assert result.is_https is True

    def test_validate_link_http(self, validator: LinkValidator) -> None:
        """Test validating HTTP link."""
        result = validator.validate_link("http://example.com")

        assert result.is_valid is True
        assert result.is_https is False
        assert len(result.issues) > 0  # Should warn about HTTP

    def test_validate_link_with_tracking(
        self, validator: LinkValidator
    ) -> None:
        """Test link with tracking parameters."""
        result = validator.validate_link(
            "https://example.com?utm_source=test"
        )

        assert result.has_tracking is True
        assert "tracking" in " ".join(result.issues).lower()

    def test_validate_link_invalid_scheme(
        self, validator: LinkValidator
    ) -> None:
        """Test link with invalid scheme."""
        result = validator.validate_link("ftp://example.com")

        assert result.is_valid is False

    def test_validate_link_empty_raises_error(
        self, validator: LinkValidator
    ) -> None:
        """Test validating empty URL raises error."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            validator.validate_link("")

    def test_validate_link_malformed(self, validator: LinkValidator) -> None:
        """Test malformed URL."""
        result = validator.validate_link("not-a-valid-url")

        assert result.is_valid is False
        assert len(result.issues) > 0

    def test_validate_link_with_path(self, validator: LinkValidator) -> None:
        """Test link with path."""
        result = validator.validate_link("https://example.com/path/to/page")

        assert result.is_valid is True

    def test_validate_link_with_query(self, validator: LinkValidator) -> None:
        """Test link with query parameters."""
        result = validator.validate_link("https://example.com?key=value")

        assert result.is_valid is True