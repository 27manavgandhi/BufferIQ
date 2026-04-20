"""Tests for model registry."""

import tempfile
from pathlib import Path

import joblib
import pytest

from bufferiq.ml.training.model_registry import ModelRegistry


class TestModelRegistry:
    """Test model registry."""

    @pytest.fixture
    def temp_dir(self) -> str:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def registry(self, temp_dir: str) -> ModelRegistry:
        """Create model registry."""
        return ModelRegistry(registry_dir=temp_dir)

    @pytest.fixture
    def mock_model_path(self, temp_dir: str) -> str:
        """Create mock model file."""
        model_path = Path(temp_dir) / "test_model.joblib"
        joblib.dump({"model": "data"}, model_path)
        return str(model_path)

    def test_init(self, registry: ModelRegistry) -> None:
        """Test initialization."""
        assert registry.registry_dir.exists()
        assert registry.models_dir.exists()
        assert registry.registry_file.exists()

    def test_register_model(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test registering a model."""
        model_id = registry.register_model(
            model_path=mock_model_path,
            version="1.0.0",
            metrics={"r2": 0.85, "mae": 2.3},
            metadata={"features": ["f1", "f2"]},
        )

        assert model_id == "model_1_0_0"
        assert model_id in registry.registry["models"]

    def test_register_model_as_production(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test registering model as production."""
        model_id = registry.register_model(
            model_path=mock_model_path,
            version="1.0.0",
            metrics={"r2": 0.85},
            metadata={},
            is_production=True,
        )

        assert registry.registry["production_model"] == model_id
        assert registry.registry["models"][model_id]["is_production"]

    def test_get_model_by_id(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test getting model by ID."""
        model_id = registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})

        model = registry.get_model(model_id=model_id)

        assert model["model_id"] == model_id
        assert model["version"] == "1.0.0"

    def test_get_model_by_version(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test getting model by version."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})

        model = registry.get_model(version="1.0.0")

        assert model["version"] == "1.0.0"

    def test_get_model_not_found(self, registry: ModelRegistry) -> None:
        """Test getting non-existent model raises error."""
        with pytest.raises(ValueError, match="Model not found"):
            registry.get_model(model_id="nonexistent")

    def test_get_production_model(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test getting production model."""
        registry.register_model(
            mock_model_path, "1.0.0", {"r2": 0.85}, {}, is_production=True
        )

        model = registry.get_model(production_only=True)

        assert model["is_production"]

    def test_get_production_model_not_set(self, registry: ModelRegistry) -> None:
        """Test getting production model when none set raises error."""
        with pytest.raises(ValueError, match="No production model"):
            registry.get_model(production_only=True)

    def test_load_model(self, registry: ModelRegistry, mock_model_path: str) -> None:
        """Test loading model."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})

        model = registry.load_model(version="1.0.0")

        assert model == {"model": "data"}

    def test_promote_to_production(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test promoting model to production."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})

        registry.promote_to_production("1.0.0")

        assert registry.registry["production_model"] == "model_1_0_0"
        assert registry.registry["models"]["model_1_0_0"]["is_production"]

    def test_promote_to_production_demotes_current(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test promoting demotes current production model."""
        # Register and promote first model
        registry.register_model(
            mock_model_path, "1.0.0", {"r2": 0.85}, {}, is_production=True
        )

        # Register and promote second model
        registry.register_model(mock_model_path, "2.0.0", {"r2": 0.90}, {})
        registry.promote_to_production("2.0.0")

        # First model should be demoted
        assert not registry.registry["models"]["model_1_0_0"]["is_production"]
        assert registry.registry["models"]["model_2_0_0"]["is_production"]

    def test_list_models(self, registry: ModelRegistry, mock_model_path: str) -> None:
        """Test listing models."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})
        registry.register_model(mock_model_path, "2.0.0", {"r2": 0.90}, {})

        models = registry.list_models()

        assert len(models) == 2

    def test_list_models_production_only(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test listing only production models."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})
        registry.register_model(
            mock_model_path, "2.0.0", {"r2": 0.90}, {}, is_production=True
        )

        models = registry.list_models(production_only=True)

        assert len(models) == 1
        assert models[0]["version"] == "2.0.0"

    def test_compare_models(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test comparing models."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})
        registry.register_model(mock_model_path, "2.0.0", {"r2": 0.90}, {})

        comparison = registry.compare_models(
            ["model_1_0_0", "model_2_0_0"], metric="r2"
        )

        assert len(comparison) == 2
        assert "r2" in comparison.columns

    def test_get_best_model(
        self, registry: ModelRegistry, mock_model_path: str
    ) -> None:
        """Test getting best model."""
        registry.register_model(mock_model_path, "1.0.0", {"r2": 0.85}, {})
        registry.register_model(mock_model_path, "2.0.0", {"r2": 0.90}, {})

        best = registry.get_best_model(metric="r2", higher_is_better=True)

        assert best["version"] == "2.0.0"
        assert best["metrics"]["r2"] == 0.90

    def test_get_best_model_no_models(self, registry: ModelRegistry) -> None:
        """Test getting best model with no models raises error."""
        with pytest.raises(ValueError, match="No models registered"):
            registry.get_best_model()
