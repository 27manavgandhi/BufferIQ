"""Tests for optimization configuration schema."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bufferiq.ml.optimization.config_schema import OptimizationConfig


def test_config_valid_grid_search():
    """Test valid grid search configuration."""
    config = OptimizationConfig(
        model_type="xgboost",
        strategy="grid",
        cv_folds=5,
        scoring="r2",
    )
    assert config.model_type == "xgboost"
    assert config.strategy == "grid"
    assert config.cv_folds == 5


def test_config_valid_random_search():
    """Test valid random search configuration."""
    config = OptimizationConfig(
        model_type="lightgbm",
        strategy="random",
        n_iter=50,
        cv_folds=5,
    )
    assert config.strategy == "random"
    assert config.n_iter == 50


def test_config_valid_bayesian_search():
    """Test valid Bayesian search configuration."""
    config = OptimizationConfig(
        model_type="random_forest",
        strategy="bayesian",
        n_iter=30,
        cv_folds=5,
    )
    assert config.strategy == "bayesian"
    assert config.n_iter == 30


def test_config_invalid_model_type():
    """Test invalid model type raises error."""
    with pytest.raises(ValidationError, match="Invalid model_type"):
        OptimizationConfig(
            model_type="invalid_model",
            strategy="grid",
        )


def test_config_invalid_strategy():
    """Test invalid strategy raises error."""
    with pytest.raises(ValidationError, match="Invalid strategy"):
        OptimizationConfig(
            model_type="xgboost",
            strategy="invalid_strategy",
        )


def test_config_cv_folds_too_small():
    """Test cv_folds < 2 raises error."""
    with pytest.raises(ValidationError):
        OptimizationConfig(
            model_type="xgboost",
            strategy="grid",
            cv_folds=1,
        )


def test_config_cv_folds_too_large():
    """Test cv_folds > 10 raises error."""
    with pytest.raises(ValidationError):
        OptimizationConfig(
            model_type="xgboost",
            strategy="grid",
            cv_folds=15,
        )


def test_config_random_requires_n_iter():
    """Test random search requires n_iter."""
    with pytest.raises(ValidationError, match="n_iter is required"):
        OptimizationConfig(
            model_type="xgboost",
            strategy="random",
        )


def test_config_bayesian_requires_n_iter():
    """Test Bayesian search requires n_iter."""
    with pytest.raises(ValidationError, match="n_iter is required"):
        OptimizationConfig(
            model_type="xgboost",
            strategy="bayesian",
        )


def test_config_defaults():
    """Test configuration defaults."""
    config = OptimizationConfig(
        model_type="xgboost",
        strategy="grid",
    )
    assert config.cv_folds == 5
    assert config.scoring == "r2"
    assert config.n_jobs == -1
    assert config.random_state == 42
    assert config.output_dir == Path("outputs/optimizations")


def test_config_from_yaml(tmp_path):
    """Test loading configuration from YAML."""
    config_path = tmp_path / "config.yaml"
    config_content = """
model_type: xgboost
strategy: grid
cv_folds: 5
scoring: r2
"""
    config_path.write_text(config_content)
    
    config = OptimizationConfig.from_yaml(config_path)
    assert config.model_type == "xgboost"
    assert config.strategy == "grid"


def test_config_from_yaml_file_not_found():
    """Test from_yaml raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        OptimizationConfig.from_yaml(Path("nonexistent.yaml"))


def test_config_to_yaml(tmp_path):
    """Test saving configuration to YAML."""
    config = OptimizationConfig(
        model_type="xgboost",
        strategy="grid",
        cv_folds=5,
    )
    
    output_path = tmp_path / "output.yaml"
    config.to_yaml(output_path)
    
    assert output_path.exists()
    loaded_config = OptimizationConfig.from_yaml(output_path)
    assert loaded_config.model_type == config.model_type