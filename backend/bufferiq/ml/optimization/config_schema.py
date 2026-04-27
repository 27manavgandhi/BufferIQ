"""Pydantic schema for optimization configuration."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, validator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class OptimizationConfig(BaseModel):
    """
    Configuration schema for hyperparameter optimization.

    Validates optimization settings and provides defaults.
    """

    model_type: str = Field(
        ...,
        description="Model type to optimize",
    )
    strategy: str = Field(
        ...,
        description="Search strategy (grid, random, bayesian)",
    )
    search_space: Optional[dict[str, Any]] = Field(
        None,
        description="Custom search space (uses default if not provided)",
    )
    cv_folds: int = Field(
        5,
        description="Number of cross-validation folds",
        ge=2,
        le=10,
    )
    scoring: str = Field(
        "r2",
        description="Scoring metric for optimization",
    )
    n_iter: Optional[int] = Field(
        None,
        description="Number of iterations (for random/bayesian)",
        ge=1,
    )
    n_jobs: int = Field(
        -1,
        description="Number of parallel jobs (-1 for all cores)",
    )
    random_state: int = Field(
        42,
        description="Random seed for reproducibility",
    )
    early_stopping_rounds: Optional[int] = Field(
        None,
        description="Stop if no improvement for N rounds",
        ge=1,
    )
    output_dir: Path = Field(
        Path("outputs/optimizations"),
        description="Directory for optimization results",
    )

    @validator("model_type")
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        valid_types = ["xgboost", "lightgbm", "random_forest"]
        if v not in valid_types:
            raise ValueError(f"Invalid model_type: {v}. Supported: {valid_types}")
        return v

    @validator("strategy")
    def validate_strategy(cls, v: str) -> str:
        """Validate search strategy."""
        valid_strategies = ["grid", "random", "bayesian"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Supported: {valid_strategies}")
        return v

    @validator("n_iter")
    def validate_n_iter(cls, v: Optional[int], values: dict[str, Any]) -> Optional[int]:
        """Validate n_iter is provided for random/bayesian."""
        strategy = values.get("strategy")
        if strategy in ["random", "bayesian"] and v is None:
            raise ValueError(f"n_iter is required for {strategy} search")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "OptimizationConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Validated configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid

        Example:
            >>> config = OptimizationConfig.from_yaml(
            ...     Path("configs/optimization/xgboost_grid.yaml")
            ... )
            >>> print(config.model_type)
            xgboost
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save config file
        """
        data = self.dict()
        # Convert Path to string for YAML serialization
        data["output_dir"] = str(data["output_dir"])

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved config to {path}")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
