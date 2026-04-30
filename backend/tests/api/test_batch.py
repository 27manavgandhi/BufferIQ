"""Tests for batch prediction endpoint."""

import pytest


def test_batch_predict_success(client):
    """Test successful batch prediction."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {
                    "content": "First post",
                    "platform": "linkedin",
                },
            },
            {
                "id": "post_2",
                "request": {
                    "content": "Second post",
                    "platform": "twitter",
                },
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert "metadata" in data
    assert len(data["predictions"]) == 2


def test_batch_predict_empty(client):
    """Test batch with empty items."""
    request = {"items": []}

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 422


def test_batch_predict_too_large(client):
    """Test batch exceeding size limit."""
    request = {
        "items": [
            {
                "id": f"post_{i}",
                "request": {
                    "content": f"Post {i}",
                    "platform": "linkedin",
                },
            }
            for i in range(101)  # Exceeds 100 limit
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 422


def test_batch_predict_single_item(client):
    """Test batch with single item."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {
                    "content": "Single post",
                    "platform": "linkedin",
                },
            }
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == 1


def test_batch_predict_metadata(client):
    """Test batch prediction metadata."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {
                    "content": "Post 1",
                    "platform": "linkedin",
                },
            },
            {
                "id": "post_2",
                "request": {
                    "content": "Post 2",
                    "platform": "twitter",
                },
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    data = response.json()
    metadata = data["metadata"]

    assert "total_items" in metadata
    assert "processing_time_ms" in metadata
    assert "cache_hits" in metadata
    assert "errors" in metadata

    assert metadata["total_items"] == 2
    assert metadata["processing_time_ms"] > 0


def test_batch_predict_mixed_platforms(client):
    """Test batch with different platforms."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {"content": "LinkedIn", "platform": "linkedin"},
            },
            {
                "id": "post_2",
                "request": {"content": "Twitter", "platform": "twitter"},
            },
            {
                "id": "post_3",
                "request": {"content": "Bluesky", "platform": "bluesky"},
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == 3


def test_batch_predict_preserves_ids(client):
    """Test batch preserves request IDs."""
    request = {
        "items": [
            {
                "id": "custom_id_1",
                "request": {"content": "Post 1", "platform": "linkedin"},
            },
            {
                "id": "custom_id_2",
                "request": {"content": "Post 2", "platform": "twitter"},
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    data = response.json()

    ids = [pred["id"] for pred in data["predictions"]]
    assert "custom_id_1" in ids
    assert "custom_id_2" in ids


def test_batch_predict_duplicate_ids(client):
    """Test batch with duplicate IDs."""
    request = {
        "items": [
            {
                "id": "same_id",
                "request": {"content": "Post 1", "platform": "linkedin"},
            },
            {
                "id": "same_id",
                "request": {"content": "Post 2", "platform": "twitter"},
            },
        ]
    }

    # Should still work - IDs don't need to be unique
    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200


def test_batch_predict_with_scheduled_times(client):
    """Test batch with scheduled times."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {
                    "content": "Morning post",
                    "platform": "linkedin",
                    "scheduled_time": "2026-04-30T09:00:00Z",
                },
            },
            {
                "id": "post_2",
                "request": {
                    "content": "Evening post",
                    "platform": "linkedin",
                    "scheduled_time": "2026-04-30T18:00:00Z",
                },
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200


def test_batch_predict_with_media_flags(client):
    """Test batch with media flags."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {
                    "content": "With image",
                    "platform": "linkedin",
                    "has_media": True,
                },
            },
            {
                "id": "post_2",
                "request": {
                    "content": "With link",
                    "platform": "twitter",
                    "has_link": True,
                },
            },
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200


def test_batch_predict_response_structure(client):
    """Test batch response structure."""
    request = {
        "items": [
            {
                "id": "post_1",
                "request": {"content": "Test", "platform": "linkedin"},
            }
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    data = response.json()

    prediction = data["predictions"][0]
    assert "id" in prediction
    assert "prediction" in prediction or "error" in prediction


def test_batch_predict_large_batch(client):
    """Test batch with maximum allowed items."""
    request = {
        "items": [
            {
                "id": f"post_{i}",
                "request": {
                    "content": f"Post {i}",
                    "platform": "linkedin",
                },
            }
            for i in range(100)  # Maximum allowed
        ]
    }

    response = client.post("/api/v1/batch/predict", json=request)
    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == 100