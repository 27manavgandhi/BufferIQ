"""Tests for hashtags API router."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from bufferiq.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHashtagsRouter:
    """Test hashtags router endpoints."""

    def test_analyze_endpoint(self, client):
        """Test analyze endpoint."""
        with patch(
            "bufferiq.api.dependencies.hashtags.get_hashtag_service"
        ) as mock_service:
            # Mock service
            mock_instance = Mock()
            mock_instance.analyze_hashtag = AsyncMock(
                return_value={
                    "hashtag": "ai",
                    "platform": "linkedin",
                    "performance": {
                        "total_uses": 100,
                        "avg_engagement": 150.0,
                        "engagement_lift": 0.25,
                        "trend_direction": "growing",
                        "roi": 5.0,
                    },
                    "risk": {
                        "risk_level": "none",
                        "is_safe": True,
                        "reasons": [],
                        "recommendation": "use",
                    },
                    "related": {
                        "synonyms": [],
                        "complementary": [],
                    },
                }
            )
            mock_service.return_value = mock_instance

            response = client.post(
                "/api/v1/hashtags/analyze",
                json={
                    "hashtag": "ai",
                    "platform": "linkedin",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["hashtag"] == "ai"
            assert data["platform"] == "linkedin"

    def test_recommend_endpoint(self, client):
        """Test recommend endpoint."""
        with patch(
            "bufferiq.api.dependencies.hashtags.get_hashtag_service"
        ) as mock_service:
            mock_instance = Mock()
            mock_instance.recommend_hashtags = AsyncMock(
                return_value=["ai", "tech", "innovation"]
            )
            mock_service.return_value = mock_instance

            response = client.post(
                "/api/v1/hashtags/recommend",
                json={
                    "content": "AI insights",
                    "platform": "linkedin",
                    "count": 3,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "hashtags" in data
            assert len(data["hashtags"]) == 3

    def test_validate_endpoint(self, client):
        """Test validate endpoint."""
        with patch(
            "bufferiq.api.dependencies.hashtags.get_hashtag_service"
        ) as mock_service:
            from bufferiq.ml.hashtags.risks.detector import HashtagRisk

            mock_instance = Mock()
            mock_instance.validate_hashtags = AsyncMock(
                return_value={
                    "ai": HashtagRisk(
                        hashtag="ai",
                        risk_level="none",
                        recommendation="use",
                    ),
                }
            )
            mock_service.return_value = mock_instance

            response = client.post(
                "/api/v1/hashtags/validate",
                json={
                    "hashtags": ["ai"],
                    "platform": "linkedin",
                },
            )

            assert response.status_code == 200

    def test_invalid_platform(self, client):
        """Test with invalid platform."""
        response = client.post(
            "/api/v1/hashtags/analyze",
            json={
                "hashtag": "ai",
                "platform": "invalid",
            },
        )

        # Should fail validation
        assert response.status_code == 422