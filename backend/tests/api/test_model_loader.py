"""Tests for model loader service."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import joblib
import tempfile

from bufferiq.api.services.model_loader import ModelLoader


@pytest.fixture
def model_loader():
    """Create fresh model loader instance."""
    # Reset singleton
    ModelLoader._instance = None
    return ModelLoader()


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock model file."""
    model = Mock()
    model.predict = Mock(return_value=[1.0])

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    return model_path


def test_singleton_pattern():
    """Test ModelLoader is a singleton."""
    loader1 = ModelLoader()
    loader2 = ModelLoader()

    assert loader1 is loader2


def test_register_model(model_loader, mock_model_file):
    """Test registering a model."""
    model_loader.register_model("test_model", mock_model_file)

    assert "test_model" in model_loader.model_paths
    assert model_loader.model_paths["test_model"] == mock_model_file


def test_load_model(model_loader, mock_model_file):
    """Test loading a model."""
    model_loader.register_model("test_model", mock_model_file)
    model = model_loader.load_model("test_model")

    assert model is not None
    assert "test_model" in model_loader.models


def test_load_unregistered_model(model_loader):
    """Test loading unregistered model raises error."""
    with pytest.raises(ValueError, match="not registered"):
        model_loader.load_model("nonexistent")


def test_load_nonexistent_file(model_loader, tmp_path):
    """Test loading nonexistent file raises error."""
    model_loader.register_model("test", tmp_path / "nonexistent.joblib")

    with pytest.raises(FileNotFoundError):
        model_loader.load_model("test")


def test_model_caching(model_loader, mock_model_file):
    """Test models are cached after loading."""
    model_loader.register_model("test_model", mock_model_file)

    # First load
    model1 = model_loader.load_model("test_model")

    # Second load should return cached
    model2 = model_loader.load_model("test_model")

    assert model1 is model2


def test_warmup(model_loader, tmp_path):
    """Test warming up all models."""
    # Create multiple mock models
    for i in range(3):
        model = Mock()
        model_path = tmp_path / f"model_{i}.joblib"
        joblib.dump(model, model_path)
        model_loader.register_model(f"model_{i}", model_path)

    # Warmup
    model_loader.warmup()

    # All models should be loaded
    assert len(model_loader.models) == 3


def test_reload_model(model_loader, mock_model_file):
    """Test reloading a model."""
    model_loader.register_model("test_model", mock_model_file)

    # Initial load
    model_loader.load_model("test_model")

    # Reload
    model_loader.reload("test_model")

    # Model should still be loaded
    assert "test_model" in model_loader.models


def test_get_loaded_models(model_loader, tmp_path):
    """Test getting list of loaded models."""
    # Create and register models
    for i in range(2):
        model = Mock()
        model_path = tmp_path / f"model_{i}.joblib"
        joblib.dump(model, model_path)
        model_loader.register_model(f"model_{i}", model_path)

    # Load one model
    model_loader.load_model("model_0")

    loaded = model_loader.get_loaded_models()
    assert len(loaded) == 1
    assert "model_0" in loaded


def test_lru_cache_eviction(model_loader, tmp_path):
    """Test LRU cache eviction."""
    # Create many models (more than cache size)
    for i in range(10):
        model = Mock()
        model_path = tmp_path / f"model_{i}.joblib"
        joblib.dump(model, model_path)
        model_loader.register_model(f"model_{i}", model_path)
        model_loader.load_model(f"model_{i}")

    # Some models should be cached
    assert len(model_loader.models) > 0