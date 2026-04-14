"""Model registry for version control."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Version control for trained models."""

    def __init__(self, registry_dir: str = "outputs/models/registry") -> None:
        """
        Initialize model registry.

        Args:
            registry_dir: Directory for model registry

        Registry structure:
            outputs/models/registry/
                ├── registry.json  # Metadata for all models
                └── models/
                    ├── model_v1_0_0.joblib
                    ├── model_v1_1_0.joblib
                    └── model_v2_0_0.joblib
        """
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.registry_file = self.registry_dir / "registry.json"

        # Create directories
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load registry
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry: dict[str, Any] = {"models": {}, "production_model": None}
            self._save_registry()

    def register_model(
        self,
        model_path: str,
        version: str,
        metrics: dict[str, float],
        metadata: dict[str, Any],
        is_production: bool = False,
    ) -> str:
        """
        Register a trained model.

        Args:
            model_path: Path to saved model file
            version: Semantic version (e.g., "1.0.0")
            metrics: Performance metrics
            metadata: Additional info (hyperparameters, features, etc.)
            is_production: Mark as production model

        Returns:
            Model ID

        Example:
            >>> registry = ModelRegistry()
            >>> model_id = registry.register_model(
            ...     "model.joblib",
            ...     "1.0.0",
            ...     {"r2": 0.85, "mae": 2.3},
            ...     {"features": ["f1", "f2"]}
            ... )
        """
        # Generate model ID
        model_id = f"model_{version.replace('.', '_')}"

        # Copy model to registry
        dest_path = self.models_dir / f"{model_id}.joblib"
        shutil.copy(model_path, dest_path)

        # Create model entry
        model_entry = {
            "model_id": model_id,
            "version": version,
            "path": str(dest_path),
            "metrics": metrics,
            "metadata": metadata,
            "is_production": is_production,
            "registered_at": datetime.now().isoformat(),
        }

        # Add to registry
        self.registry["models"][model_id] = model_entry

        # Set as production if requested
        if is_production:
            self._set_production(model_id)

        # Save registry
        self._save_registry()

        logger.info(f"Registered model: {model_id} (version {version})")

        return model_id

    def get_model(
        self,
        version: Optional[str] = None,
        model_id: Optional[str] = None,
        production_only: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve model metadata.

        Args:
            version: Model version
            model_id: Model ID
            production_only: Only return production model

        Returns:
            Model metadata dict

        Raises:
            ValueError: If no model found matching criteria
        """
        # Priority: model_id > version > production_only > latest
        if model_id:
            if model_id not in self.registry["models"]:
                raise ValueError(f"Model not found: {model_id}")
            return self.registry["models"][model_id]

        if version:
            version_id = f"model_{version.replace('.', '_')}"
            if version_id not in self.registry["models"]:
                raise ValueError(f"Model version not found: {version}")
            return self.registry["models"][version_id]

        if production_only:
            prod_id = self.registry.get("production_model")
            if not prod_id:
                raise ValueError("No production model set")
            return self.registry["models"][prod_id]

        # Return latest model
        if not self.registry["models"]:
            raise ValueError("No models registered")

        models = list(self.registry["models"].values())
        models.sort(key=lambda x: x["registered_at"], reverse=True)

        return models[0]

    def load_model(
        self, version: Optional[str] = None, model_id: Optional[str] = None
    ) -> Any:
        """
        Load model from registry.

        Args:
            version: Model version
            model_id: Model ID

        Returns:
            Loaded model object

        Example:
            >>> registry = ModelRegistry()
            >>> model = registry.load_model(version="1.0.0")
        """
        model_info = self.get_model(version=version, model_id=model_id)
        model_path = model_info["path"]

        model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_info['model_id']}")

        return model

    def promote_to_production(self, version: str) -> None:
        """
        Promote model to production.

        Args:
            version: Version to promote

        Raises:
            ValueError: If version not found
        """
        model_id = f"model_{version.replace('.', '_')}"

        if model_id not in self.registry["models"]:
            raise ValueError(f"Model version not found: {version}")

        # Demote current production
        current_prod = self.registry.get("production_model")
        if current_prod and current_prod in self.registry["models"]:
            self.registry["models"][current_prod]["is_production"] = False

        # Promote new model
        self._set_production(model_id)

        logger.info(f"Promoted {version} to production")

    def list_models(
        self, production_only: bool = False, sort_by: str = "registered_at"
    ) -> list[dict[str, Any]]:
        """
        List all registered models.

        Args:
            production_only: Only show production models
            sort_by: Field to sort by

        Returns:
            List of model metadata dicts
        """
        models = list(self.registry["models"].values())

        if production_only:
            models = [m for m in models if m.get("is_production", False)]

        # Sort
        models.sort(key=lambda x: x.get(sort_by, ""), reverse=True)

        return models

    def compare_models(self, model_ids: list[str], metric: str = "r2") -> pd.DataFrame:
        """
        Compare multiple models on a metric.

        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare

        Returns:
            DataFrame with comparison

        Example:
            >>> registry = ModelRegistry()
            >>> comparison = registry.compare_models(
            ...     ["model_1_0_0", "model_1_1_0"],
            ...     metric="r2"
            ... )
        """
        comparison_data = []

        for model_id in model_ids:
            if model_id in self.registry["models"]:
                model = self.registry["models"][model_id]
                comparison_data.append(
                    {
                        "model_id": model_id,
                        "version": model["version"],
                        metric: model["metrics"].get(metric, None),
                        "registered_at": model["registered_at"],
                        "is_production": model.get("is_production", False),
                    }
                )

        df = pd.DataFrame(comparison_data)

        if not df.empty and metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def get_best_model(
        self, metric: str = "r2", higher_is_better: bool = True
    ) -> dict[str, Any]:
        """
        Get best model based on metric.

        Args:
            metric: Metric to optimize
            higher_is_better: Whether higher metric is better

        Returns:
            Best model metadata

        Raises:
            ValueError: If no models found
        """
        models = list(self.registry["models"].values())

        if not models:
            raise ValueError("No models registered")

        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.get("metrics", {})]

        if not models_with_metric:
            raise ValueError(f"No models have metric: {metric}")

        # Sort by metric
        models_with_metric.sort(
            key=lambda x: x["metrics"][metric], reverse=higher_is_better
        )

        return models_with_metric[0]

    def _set_production(self, model_id: str) -> None:
        """Set model as production."""
        self.registry["production_model"] = model_id
        self.registry["models"][model_id]["is_production"] = True

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
