"""
Tests for content analysis API router.
"""

import pytest
from fastapi.testclient import TestClient

from bufferiq.api.app import app

client = TestClient(app)


class TestContentRouter:
    """Test content analysis API endpoints."""

    def test_analyze_content_basic(self) -> None:
        """Test basic content analysis endpoint."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Great post about AI!",
                "platform": "linkedin",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "platform" in data

    def test_analyze_content_invalid_platform(self) -> None:
        """Test invalid platform returns error."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Test post",
                "platform": "facebook",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_content_empty_text(self) -> None:
        """Test empty text returns error."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "",
                "platform": "linkedin",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_content_with_user_id(self) -> None:
        """Test analysis with user ID."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Test post",
                "platform": "linkedin",
                "user_id": "user123",
            },
        )

        assert response.status_code == 200

    def test_analyze_content_without_optimization(self) -> None:
        """Test analysis without optimization."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Test post",
                "platform": "linkedin",
                "include_optimization": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "optimization" not in data or data["optimization"] is None

    def test_batch_analyze_basic(self) -> None:
        """Test batch analysis endpoint."""
        response = client.post(
            "/api/v1/content/batch",
            json={
                "posts": [
                    {"text": "First post"},
                    {"text": "Second post"},
                ],
                "platform": "linkedin",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_analyzed" in data

    def test_batch_analyze_empty_posts(self) -> None:
        """Test batch analysis with empty posts."""
        response = client.post(
            "/api/v1/content/batch",
            json={
                "posts": [],
                "platform": "linkedin",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_batch_analyze_invalid_platform(self) -> None:
        """Test batch analysis with invalid platform."""
        response = client.post(
            "/api/v1/content/batch",
            json={
                "posts": [{"text": "Test"}],
                "platform": "facebook",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_optimize_content_basic(self) -> None:
        """Test content optimization endpoint."""
        response = client.post(
            "/api/v1/content/optimize",
            json={
                "text": "Test post",
                "platform": "linkedin",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should return optimization data
        assert isinstance(data, dict)

    def test_optimize_content_invalid_platform(self) -> None:
        """Test optimization with invalid platform."""
        response = client.post(
            "/api/v1/content/optimize",
            json={
                "text": "Test post",
                "platform": "facebook",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_content_linkedin(self) -> None:
        """Test LinkedIn content analysis."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Professional post about business.",
                "platform": "linkedin",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "linkedin"

    def test_analyze_content_twitter(self) -> None:
        """Test Twitter content analysis."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Quick tweet!",
                "platform": "twitter",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "twitter"

    def test_analyze_content_bluesky(self) -> None:
        """Test Bluesky content analysis."""
        response = client.post(
            "/api/v1/content/analyze",
            json={
                "text": "Bluesky post here.",
                "platform": "bluesky",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "bluesky"