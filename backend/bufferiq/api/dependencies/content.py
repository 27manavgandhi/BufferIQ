"""
Content analysis dependencies.

Dependency injection for content endpoints.
"""

from typing import Generator

from bufferiq.ml.content.intelligence.service import ContentIntelligenceService


def get_content_service() -> Generator[ContentIntelligenceService, None, None]:
    """
    Get content intelligence service.

    Yields:
        ContentIntelligenceService instance
    """
    service = ContentIntelligenceService()
    try:
        yield service
    finally:
        pass  # Cleanup if needed