"""Pydantic schemas for ensemble configuration."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class MetaLearnerConfig(BaseModel):
    """Configuration for meta-learner."""

    type: Literal["ridge", "lasso", "elasticnet", "xgboost", "lightgbm"] = Field(
        default="ridge", description="Type of meta-learner"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Meta-learner hyperparameters"
    )


class OptunaConfig(BaseModel):
    """Configuration for Optuna weight optimization."""

    n_trials: int = Field(default=100, ge=1, description="Number of Optuna trials")
    timeout: Optional[int] = Field(default=600, ge=1, description="Timeout in seconds")
    sampler: Literal["tpe", "random", "grid"] = Field(
        default="tpe", description="Optuna sampler"
    )
    cv_folds: int = Field(default=5, ge=2, description="Cross-validation folds")


class SelectionConfig(BaseModel):
    """Configuration for model selection."""

    min_performance: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum R² to include model"
    )
    max_models: int = Field(default=5, ge=1, description="Maximum models in ensemble")
    min_diversity: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Minimum diversity required"
    )


class EnsembleConfig(BaseModel):
    """Main ensemble configuration."""

    ensemble_type: Literal[
        "voting", "stacking", "blending", "weighted_average", "auto"
    ] = Field(description="Type of ensemble to build")

    base_models: list[str] = Field(description="Paths to base model files")

    voting_method: Optional[Literal["soft", "hard"]] = Field(
        default="soft", description="Voting method (for voting ensemble)"
    )

    meta_learner: Optional[MetaLearnerConfig] = Field(
        default=None, description="Meta-learner config (for stacking/blending)"
    )

    blend_split: Optional[float] = Field(
        default=0.3, ge=0.1, le=0.5, description="Blend split fraction (for blending)"
    )

    cv_folds: int = Field(default=5, ge=2, description="Cross-validation folds")

    passthrough: bool = Field(
        default=False, description="Include original features in meta-learner"
    )

    weights: Optional[list[float]] = Field(
        default=None, description="Custom weights for models"
    )

    weight_optimization: Literal[
        "none", "uniform", "performance", "optuna", "grid"
    ] = Field(default="performance", description="Method for weight optimization")

    optuna_config: Optional[OptunaConfig] = Field(
        default=None, description="Optuna configuration"
    )

    selection: Optional[SelectionConfig] = Field(
        default=None, description="Model selection configuration"
    )

    diversity_threshold: Optional[float] = Field(
        default=0.15, ge=0.0, le=1.0, description="Minimum diversity threshold"
    )

    output_dir: str = Field(
        default="outputs/models/ensembles", description="Output directory for ensemble"
    )

    model_name: str = Field(description="Name for saved ensemble model")

    version: Optional[str] = Field(default="1.0.0", description="Ensemble version")

    description: Optional[str] = Field(default=None, description="Ensemble description")

    random_state: int = Field(default=42, description="Random seed for reproducibility")

    @field_validator("base_models")
    @classmethod
    def validate_base_models(cls, v: list[str]) -> list[str]:
        """Validate base_models is not empty."""
        if not v:
            raise ValueError("base_models cannot be empty")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Validate weights if provided."""
        if v is not None:
            if not all(w >= 0 for w in v):
                raise ValueError("All weights must be non-negative")

            total = sum(v)
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total}")

        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"
