"""Model loading and caching service."""

import joblib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from sklearn.base import BaseEstimator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Singleton model loader with LRU caching.

    Loads models lazily and caches them in memory.
    Limits memory usage through LRU eviction.
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls) -> "ModelLoader":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize model loader."""
        if self._initialized:
            return

        self.models: Dict[str, BaseEstimator] = {}
        self.model_paths: Dict[str, Path] = {}
        self._initialized = True

        logger.info("ModelLoader initialized")

    def register_model(self, name: str, path: Path) -> None:
        """
        Register a model path.

        Args:
            name: Model identifier
            path: Path to model file
        """
        self.model_paths[name] = Path(path)
        logger.debug(f"Registered model: {name} -> {path}")

    @lru_cache(maxsize=5)
    def _load_from_disk(self, path_str: str) -> BaseEstimator:
        """
        Load model from disk (cached).

        Args:
            path_str: String path to model file

        Returns:
            Loaded model
        """
        path = Path(path_str)
        logger.info(f"Loading model from {path}")
        model = joblib.load(path)
        return model

    def load_model(self, name: str) -> BaseEstimator:
        """
        Load model by name.

        Uses in-memory cache first, then LRU disk cache.

        Args:
            name: Model identifier

        Returns:
            Loaded model

        Raises:
            ValueError: If model not registered
            FileNotFoundError: If model file doesn't exist
        """
        # Check in-memory cache
        if name in self.models:
            logger.debug(f"Model {name} found in memory cache")
            return self.models[name]

        # Check if registered
        if name not in self.model_paths:
            raise ValueError(
                f"Model '{name}' not registered. "
                f"Available: {list(self.model_paths.keys())}"
            )

        # Get path
        path = self.model_paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load from disk (with LRU cache)
        model = self._load_from_disk(str(path))
        self.models[name] = model

        logger.info(f"Model {name} loaded successfully")
        return model

    def warmup(self) -> None:
        """
        Load all registered models into memory.

        Useful for startup to avoid cold start latency.
        """
        logger.info(f"Warming up {len(self.model_paths)} models...")

        for name in self.model_paths:
            try:
                self.load_model(name)
            except Exception as e:
                logger.error(f"Failed to warmup model {name}: {e}")

        logger.info(f"Warmup complete: {len(self.models)} models loaded")

    def reload(self, name: str) -> None:
        """
        Reload a specific model.

        Clears cache and loads fresh from disk.

        Args:
            name: Model identifier
        """
        # Remove from memory cache
        if name in self.models:
            del self.models[name]

        # Clear LRU cache
        self._load_from_disk.cache_clear()

        # Reload
        self.load_model(name)

        logger.info(f"Model {name} reloaded")

    def get_loaded_models(self) -> list:
        """Get list of currently loaded models."""
        return list(self.models.keys())