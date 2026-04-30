"""End-to-end integration tests."""

import pytest
from fastapi.testclient import TestClient
import time


def test_full_prediction_flow(client):
    """Test complete prediction flow."""
    # 1. Check health
    health = client.get("/health")
    assert health.status_code == 200

    # 2. List models
    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert len(models.json()["models"]) > 0

    # 3. Make prediction
    prediction = client.post(
        "/api/v1/predict",
        json={
            "content": "Integration test post",
            "platform": "linkedin",
        },
    )
    assert prediction.status_code == 200

    # 4. Verify response structure
    data = prediction.json()
    assert "engagement_score" in data
    assert "confidence" in data
    assert "breakdown" in data
    assert "metadata" in data


def test_batch_prediction_flow(client):
    """Test batch prediction flow."""
    # Make batch request
    response = client.post(
        "/api/v1/batch/predict",
        json={
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
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all predictions returned
    assert len(data["predictions"]) == 2

    # Verify metadata
    assert data["metadata"]["total_items"] == 2
    assert data["metadata"]["errors"] == 0


def test_error_handling_flow(client):
    """Test error handling."""
    # Invalid platform
    response = client.post(
        "/api/v1/predict",
        json={
            "content": "Test",
            "platform": "facebook",  # Invalid
        },
    )
    assert response.status_code == 422

    # Missing content
    response = client.post(
        "/api/v1/predict",
        json={
            "platform": "linkedin",
        },
    )
    assert response.status_code == 422


def test_caching_flow(client):
    """Test response caching."""
    request = {
        "content": "Cached test post",
        "platform": "linkedin",
    }

    # First request (cache miss)
    response1 = client.post("/api/v1/predict", json=request)
    data1 = response1.json()

    # Second request (cache hit)
    response2 = client.post("/api/v1/predict", json=request)
    data2 = response2.json()

    # Scores should be identical
    assert data1["engagement_score"] == data2["engagement_score"]


def test_multi_platform_flow(client):
    """Test predictions for all platforms."""
    platforms = ["linkedin", "twitter", "bluesky"]

    for platform in platforms:
        response = client.post(
            "/api/v1/predict",
            json={
                "content": f"Post for {platform}",
                "platform": platform,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["engagement_score"] > 0


def test_model_reload_flow(client):
    """Test model reload."""
    # Initial prediction
    response1 = client.post(
        "/api/v1/predict",
        json={
            "content": "Before reload",
            "platform": "linkedin",
        },
    )
    assert response1.status_code == 200

    # Reload model
    reload = client.post("/api/v1/models/ensemble/reload")
    assert reload.status_code == 200

    # Prediction after reload
    response2 = client.post(
        "/api/v1/predict",
        json={
            "content": "After reload",
            "platform": "linkedin",
        },
    )
    assert response2.status_code == 200


def test_performance_benchmark(client):
    """Test API performance."""
    request = {
        "content": "Performance test post",
        "platform": "linkedin",
    }

    # Warm up
    client.post("/api/v1/predict", json=request)

    # Measure 10 requests
    latencies = []
    for _ in range(10):
        start = time.time()
        response = client.post("/api/v1/predict", json=request)
        latency_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        latencies.append(latency_ms)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\nPerformance:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")

    # Should be fast (relaxed for tests)
    assert avg_latency < 1000  # 1 second average
    assert p95_latency < 2000  # 2 seconds p95


def test_concurrent_requests(client):
    """Test concurrent request handling."""
    import concurrent.futures

    def make_request(i):
        return client.post(
            "/api/v1/predict",
            json={
                "content": f"Concurrent post {i}",
                "platform": "linkedin",
            },
        )

    # Send 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        responses = [f.result() for f in futures]

    # All should succeed
    assert all(r.status_code == 200 for r in responses)


def test_health_checks_flow(client):
    """Test all health check endpoints."""
    # Main health check
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in ["healthy", "unhealthy"]

    # Readiness
    ready = client.get("/health/ready")
    assert ready.status_code in [200, 503]

    # Liveness
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    # Make some requests first
    client.post(
        "/api/v1/predict",
        json={"content": "Test", "platform": "linkedin"},
    )

    # Get metrics
    response = client.get("/metrics")
    assert response.status_code == 200

    # Should be Prometheus text format
    assert "bufferiq" in response.text