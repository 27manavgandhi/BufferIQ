"""Tests for models router."""

import pytest


def test_list_models(client):
    """Test listing available models."""
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "loaded" in data
    assert isinstance(data["models"], list)


def test_get_model_info(client):
    """Test getting model information."""
    response = client.get("/api/v1/models/ensemble")

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "path" in data
    assert "loaded" in data
    assert "exists" in data


def test_get_nonexistent_model(client):
    """Test getting info for nonexistent model."""
    response = client.get("/api/v1/models/nonexistent")

    assert response.status_code == 404


def test_reload_model(client):
    """Test reloading a model."""
    response = client.post("/api/v1/models/ensemble/reload")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "success"


def test_reload_nonexistent_model(client):
    """Test reloading nonexistent model."""
    response = client.post("/api/v1/models/nonexistent/reload")

    assert response.status_code == 500


def test_models_list_not_empty(client):
    """Test that models list is not empty."""
    response = client.get("/api/v1/models")
    data = response.json()

    assert len(data["models"]) > 0


def test_model_info_structure(client):
    """Test model info response structure."""
    response = client.get("/api/v1/models/ensemble")
    data = response.json()

    assert data["name"] == "ensemble"
    assert isinstance(data["path"], str)
    assert isinstance(data["loaded"], bool)
    assert isinstance(data["exists"], bool)