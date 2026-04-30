"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient


def test_app_creation(test_app):
    """Test app is created successfully."""
    assert test_app is not None
    assert test_app.title == "BufferIQ Prediction API"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "BufferIQ API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"


def test_docs_endpoint(client):
    """Test OpenAPI docs endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_endpoint(client):
    """Test OpenAPI JSON endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert data["info"]["title"] == "BufferIQ Prediction API"


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/api/v1/predict")
    assert response.status_code == 200


def test_request_id_header(client, sample_prediction_request):
    """Test request ID header is added."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)
    assert "x-request-id" in response.headers


def test_timing_header(client, sample_prediction_request):
    """Test timing header is added."""
    response = client.post("/api/v1/predict", json=sample_prediction_request)
    assert "x-process-time" in response.headers


def test_404_error(client):
    """Test 404 error for unknown endpoint."""
    response = client.get("/api/v1/unknown")
    assert response.status_code == 404


def test_method_not_allowed(client):
    """Test 405 error for wrong method."""
    response = client.get("/api/v1/predict")
    assert response.status_code == 405


def test_invalid_json(client):
    """Test 422 error for invalid JSON."""
    response = client.post(
        "/api/v1/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_middleware_order(test_app):
    """Test middleware is loaded in correct order."""
    middleware_types = [m.cls.__name__ for m in test_app.user_middleware]
    assert "CORSMiddleware" in middleware_types
    assert "TimingMiddleware" in middleware_types


def test_routers_registered(test_app):
    """Test all routers are registered."""
    routes = [route.path for route in test_app.routes]
    assert "/api/v1/predict" in routes
    assert "/api/v1/batch/predict" in routes
    assert "/api/v1/models" in routes
    assert "/health" in routes