"""Tests for health check endpoints."""

import pytest


def test_health_check(client):
    """Test main health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "services" in data
    assert "timestamp" in data


def test_health_services(client):
    """Test health check includes all services."""
    response = client.get("/health")
    data = response.json()

    services = data["services"]
    assert "cache" in services
    assert "models" in services


def test_readiness_check(client):
    """Test readiness probe."""
    response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "ready"


def test_liveness_check(client):
    """Test liveness probe."""
    response = client.get("/health/live")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "alive"


def test_health_timestamp_format(client):
    """Test health check timestamp format."""
    response = client.get("/health")
    data = response.json()

    # Should be ISO format timestamp
    timestamp = data["timestamp"]
    assert isinstance(timestamp, str)
    assert "T" in timestamp


def test_service_health_structure(client):
    """Test service health structure."""
    response = client.get("/health")
    data = response.json()

    for service_name, service_health in data["services"].items():
        assert "status" in service_health
        assert service_health["status"] in ["healthy", "unhealthy"]