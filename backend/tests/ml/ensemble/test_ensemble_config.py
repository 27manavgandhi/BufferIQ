"""Tests for ensemble configuration schema."""

import pytest
from pydantic import ValidationError

from bufferiq.ml.ensemble.config_schema import (
    EnsembleConfig,
    MetaLearnerConfig,
    OptunaConfig,
    SelectionConfig,
)


def test_meta_learner_config_valid():
    """Test valid meta-learner config."""
    config = MetaLearnerConfig(type="ridge", params={"alpha": 1.0})
    assert config.type == "ridge"
    assert config.params["alpha"] == 1.0


def test_meta_learner_config_defaults():
    """Test meta-learner config defaults."""
    config = MetaLearnerConfig()
    assert config.type == "ridge"
    assert config.params == {}


def test_optuna_config_valid():
    """Test valid Optuna config."""
    config = OptunaConfig(n_trials=50, timeout=300, sampler="tpe")
    assert config.n_trials == 50
    assert config.timeout == 300
    assert config.sampler == "tpe"


def test_optuna_config_defaults():
    """Test Optuna config defaults."""
    config = OptunaConfig()
    assert config.n_trials == 100
    assert config.timeout == 600
    assert config.sampler == "tpe"


def test_optuna_config_invalid_n_trials():
    """Test Optuna config with invalid n_trials."""
    with pytest.raises(ValidationError):
        OptunaConfig(n_trials=0)


def test_selection_config_valid():
    """Test valid selection config."""
    config = SelectionConfig(
        min_performance=0.8,
        max_models=3,
        min_diversity=0.2
    )
    assert config.min_performance == 0.8
    assert config.max_models == 3
    assert config.min_diversity == 0.2


def test_selection_config_defaults():
    """Test selection config defaults."""
    config = SelectionConfig()
    assert config.min_performance == 0.70
    assert config.max_models == 5
    assert config.min_diversity == 0.10


def test_ensemble_config_valid():
    """Test valid ensemble config."""
    config = EnsembleConfig(
        ensemble_type="stacking",
        base_models=["model1.joblib", "model2.joblib"],
        model_name="my_ensemble"
    )
    assert config.ensemble_type == "stacking"
    assert len(config.base_models) == 2
    assert config.model_name == "my_ensemble"


def test_ensemble_config_empty_base_models():
    """Test ensemble config with empty base_models."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        EnsembleConfig(
            ensemble_type="voting",
            base_models=[],
            model_name="test"
        )


def test_ensemble_config_invalid_ensemble_type():
    """Test ensemble config with invalid type."""
    with pytest.raises(ValidationError):
        EnsembleConfig(
            ensemble_type="invalid",
            base_models=["model1.joblib"],
            model_name="test"
        )


def test_ensemble_config_valid_weights():
    """Test ensemble config with valid weights."""
    config = EnsembleConfig(
        ensemble_type="voting",
        base_models=["model1.joblib", "model2.joblib", "model3.joblib"],
        weights=[0.5, 0.3, 0.2],
        model_name="test"
    )
    assert config.weights == [0.5, 0.3, 0.2]


def test_ensemble_config_invalid_weights_sum():
    """Test ensemble config with weights that don't sum to 1."""
    with pytest.raises(ValidationError, match="sum to 1.0"):
        EnsembleConfig(
            ensemble_type="voting",
            base_models=["model1.joblib", "model2.joblib"],
            weights=[0.6, 0.6],
            model_name="test"
        )


def test_ensemble_config_negative_weights():
    """Test ensemble config with negative weights."""
    with pytest.raises(ValidationError, match="non-negative"):
        EnsembleConfig(
            ensemble_type="voting",
            base_models=["model1.joblib", "model2.joblib"],
            weights=[0.6, -0.1, 0.5],
            model_name="test"
        )


def test_ensemble_config_defaults():
    """Test ensemble config defaults."""
    config = EnsembleConfig(
        ensemble_type="voting",
        base_models=["model1.joblib"],
        model_name="test"
    )
    assert config.voting_method == "soft"
    assert config.cv_folds == 5
    assert config.passthrough is False
    assert config.weight_optimization == "performance"
    assert config.random_state == 42


def test_ensemble_config_with_meta_learner():
    """Test ensemble config with meta-learner."""
    config = EnsembleConfig(
        ensemble_type="stacking",
        base_models=["model1.joblib", "model2.joblib"],
        meta_learner=MetaLearnerConfig(type="ridge", params={"alpha": 0.5}),
        model_name="test"
    )
    assert config.meta_learner.type == "ridge"
    assert config.meta_learner.params["alpha"] == 0.5


def test_ensemble_config_extra_fields_forbidden():
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        EnsembleConfig(
            ensemble_type="voting",
            base_models=["model1.joblib"],
            model_name="test",
            extra_field="not_allowed"
        )