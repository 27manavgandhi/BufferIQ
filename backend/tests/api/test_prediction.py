"""Tests for prediction endpoint."""

import pytest
from unittest.mock import Mock, patch


def test_predict_success(client, sample_prediction_request):
    """Test successful prediction."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)

    assert response.status_code == 200
    data = response.json()

    assert "engagement_score" in data
    assert "confidence" in data
    assert "breakdown" in data
    assert "metadata" in data

    assert data["engagement_score"] > 0
    assert 0 <= data["confidence"] <= 1


def test_predict_invalid_platform(client):
    """Test prediction with invalid platform."""
    request = {
        "content": "Test post",
        "platform": "facebook",  # Invalid
        "scheduled_time": "2026-04-30T14:00:00Z",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 422


def test_predict_missing_content(client):
    """Test prediction with missing content."""
    request = {
        "platform": "linkedin",
        "scheduled_time": "2026-04-30T14:00:00Z",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 422


def test_predict_empty_content(client):
    """Test prediction with empty content."""
    request = {
        "content": "",
        "platform": "linkedin",
        "scheduled_time": "2026-04-30T14:00:00Z",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 422


def test_predict_linkedin(client):
    """Test prediction for LinkedIn."""
    request = {
        "content": "Professional update",
        "platform": "linkedin",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_twitter(client):
    """Test prediction for Twitter."""
    request = {
        "content": "Quick tweet",
        "platform": "twitter",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_bluesky(client):
    """Test prediction for Bluesky."""
    request = {
        "content": "Bluesky post",
        "platform": "bluesky",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_with_media(client):
    """Test prediction with media."""
    request = {
        "content": "Check out this image",
        "platform": "linkedin",
        "has_media": True,
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_with_link(client):
    """Test prediction with link."""
    request = {
        "content": "Read more here",
        "platform": "linkedin",
        "has_link": True,
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_ensemble(client, sample_prediction_request):
    """Test ensemble prediction endpoint."""
    response = client.post(
        "/api/v1/predict/ensemble", json=sample_prediction_request
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["model_version"] == "ensemble"


def test_predict_breakdown_structure(client, sample_prediction_request):
    """Test breakdown structure."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)

    data = response.json()
    breakdown = data["breakdown"]

    assert "likes" in breakdown
    assert "comments" in breakdown
    assert "shares" in breakdown

    assert breakdown["likes"] >= 0
    assert breakdown["comments"] >= 0
    assert breakdown["shares"] >= 0


def test_predict_metadata_structure(client, sample_prediction_request):
    """Test metadata structure."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)

    data = response.json()
    metadata = data["metadata"]

    assert "model_version" in metadata
    assert "inference_time_ms" in metadata
    assert "features_used" in metadata
    assert "cached" in metadata
    assert "timestamp" in metadata


def test_predict_different_content_lengths(client):
    """Test predictions with different content lengths."""
    for length in [10, 100, 500, 1000]:
        request = {
            "content": "x" * length,
            "platform": "linkedin",
        }

        response = client.post("/api/v1/predict", json=request)
        assert response.status_code == 200


def test_predict_special_characters(client):
    """Test prediction with special characters."""
    request = {
        "content": "Test with émojis 🚀 and spëcial çhars!",
        "platform": "linkedin",
    }

    response = client.post("/api/v1/predict", json=request)
    assert response.status_code == 200


def test_predict_scheduled_time_formats(client):
    """Test different scheduled time formats."""
    times = [
        "2026-04-30T14:00:00Z",
        "2026-04-30T14:00:00+00:00",
        None,
    ]

    for scheduled_time in times:
        request = {
            "content": "Test post",
            "platform": "linkedin",
            "scheduled_time": scheduled_time,
        }

        response = client.post("/api/v1/predict", json=request)
        assert response.status_code == 200


def test_predict_response_time(client, sample_prediction_request):
    """Test prediction response time."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)

    assert response.status_code == 200

    # Check timing header exists
    assert "x-process-time" in response.headers

    # Extract timing
    timing_str = response.headers["x-process-time"]
    timing_ms = float(timing_str.replace("ms", ""))

    # Should be fast (< 1000ms in tests)
    assert timing_ms < 1000


def test_concurrent_predictions(client, sample_prediction_request):
    """Test concurrent predictions."""
    import concurrent.futures

    def make_request():
        return client.post("/api/v1/predict", json=sample_prediction_request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in responses)