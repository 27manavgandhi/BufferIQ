"""Training configuration schema."""

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

# Supported platforms
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class DataConfig(BaseModel):
    """Data configuration."""

    target_column: str = "engagement_rate"
    feature_columns: Optional[list[str]] = None
    platforms: list[str] = Field(default=["linkedin", "twitter", "bluesky"])
    min_samples: int = 100
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    validation_size: float = Field(default=0.1, ge=0.0, le=0.3)
    time_based_split: bool = True

    @validator("platforms")
    def validate_platforms(cls, v: list[str]) -> list[str]:
        """Validate only supported platforms."""
        invalid = [p for p in v if p not in SUPPORTED_PLATFORMS]
        if invalid:
            raise ValueError(
                f"Invalid platforms: {invalid}. Supported: {SUPPORTED_PLATFORMS}"
            )
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    model_type: Literal["xgboost", "lightgbm", "random_forest", "linear"]
    hyperparameters: dict[str, Any]
    random_state: int = 42


class TrainingConfig(BaseModel):
    """Training configuration."""

    max_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    checkpoint_monitor: str = "val_r2"
    checkpoint_mode: Literal["min", "max"] = "max"


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    experiment_name: str
    description: str
    use_cross_validation: bool = False
    cv_folds: int = Field(default=5, ge=2, le=10)


class TrainingPipelineConfig(BaseModel):
    """Complete training pipeline configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingPipelineConfig":
        """
        Load config from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            TrainingPipelineConfig instance

        Example:
            >>> config = TrainingPipelineConfig.from_yaml('configs/training/baseline.yaml')
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded config from {yaml_path}")

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save config to YAML file.

        Args:
            yaml_path: Path to save YAML config
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.dict()

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {yaml_path}")
