"""Pytest fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

from bufferiq.api.app import create_app
from bufferiq.api.services.model_loader import ModelLoader


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    model = Mock()
    model.predict = Mock(return_value=[7.5])
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    """Create a mock model loader."""
    loader = Mock(spec=ModelLoader)
    loader.models = {"ensemble": mock_model}
    loader.model_paths = {"ensemble": Mock()}
    loader.load_model = Mock(return_value=mock_model)
    loader.warmup = Mock()
    return loader


@pytest.fixture
def test_app(mock_model_loader, monkeypatch):
    """Create test FastAPI application."""
    # Override model loader
    monkeypatch.setattr(
        "bufferiq.api.app.ModelLoader",
        lambda: mock_model_loader
    )

    app = create_app()
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request."""
    return {
        "content": "Just shipped a new feature!",
        "platform": "linkedin",
        "scheduled_time": "2026-04-30T14:00:00Z",
        "has_media": False,
        "has_link": True,
    }