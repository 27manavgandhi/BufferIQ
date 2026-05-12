"""
Tests for content service.
"""

import pytest

from bufferiq.api.services.content_service import ContentService


class TestContentService:
    """Test ContentService class."""

    @pytest.fixture
    async def service(self) -> ContentService:
        """Create service fixture."""
        return ContentService()

    @pytest.mark.asyncio
    async def test_analyze_basic(self, service: ContentService) -> None:
        """Test basic analysis."""
        result = await service.analyze("Test post", "linkedin")

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_analyze_invalid_platform_raises_error(
        self, service: ContentService
    ) -> None:
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            await service.analyze("Test", "facebook")

    @pytest.mark.asyncio
    async def test_analyze_with_user_id(
        self, service: ContentService
    ) -> None:
        """Test analysis with user ID."""
        result = await service.analyze("Test", "linkedin", user_id="user123")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyze_batch_basic(
        self, service: ContentService
    ) -> None:
        """Test batch analysis."""
        posts = [{"text": "Post 1"}, {"text": "Post 2"}]
        results = await service.analyze_batch(posts, "linkedin")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_analyze_batch_invalid_platform_raises_error(
        self, service: ContentService
    ) -> None:
        """Test batch with invalid platform raises error."""
        posts = [{"text": "Test"}]
        with pytest.raises(ValueError, match="not supported"):
            await service.analyze_batch(posts, "facebook")