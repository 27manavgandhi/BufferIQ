"""
Content service layer.

Business logic for content analysis operations.
"""

from typing import Any, Dict, List, Optional

from bufferiq.ml.content.intelligence.service import ContentIntelligenceService

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ContentService:
    """
    Content service for API layer.

    Provides business logic for content operations.

    Example:
```python
        service = ContentService()
        result = await service.analyze(text, platform)
```
    """

    def __init__(self) -> None:
        """Initialize content service."""
        self.intelligence_service = ContentIntelligenceService()

    async def analyze(
        self,
        text: str,
        platform: str,
        user_id: Optional[str] = None,
        include_optimization: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze content.

        Args:
            text: Content text
            platform: Platform type
            user_id: Optional user ID
            include_optimization: Include optimization suggestions

        Returns:
            Analysis results

        Raises:
            ValueError: If validation fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        return self.intelligence_service.analyze_content(
            text=text,
            platform=platform,
            user_id=user_id,
            include_optimization=include_optimization,
        )

    async def analyze_batch(
        self, posts: List[Dict[str, Any]], platform: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple posts.

        Args:
            posts: List of posts
            platform: Platform type

        Returns:
            List of analysis results

        Raises:
            ValueError: If validation fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        return self.intelligence_service.analyze_batch(posts, platform)