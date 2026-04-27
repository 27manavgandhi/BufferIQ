"""Tests for weight optimizer."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.weight_optimizer import WeightOptimizer


@pytest.fixture
def mock_models():
    """Create mock models."""
    models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(100) + i)
        models.append(model)
    return models


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_weight_optimizer_initialization(mock_models):
    """Test weight optimizer initialization."""
    optimizer = WeightOptimizer(mock_models, method="optuna")
    assert len(optimizer.base_models) == 3
    assert optimizer.method == "optuna"


def test_weight_optimizer_empty_models():
    """Test weight optimizer with empty models."""
    with pytest.raises(ValueError, match="cannot be empty"):
        WeightOptimizer([])


def test_uniform_weights(mock_models, sample_data):
    """Test uniform weight calculation."""
    X, y = sample_data
    optimizer = WeightOptimizer(mock_models, method="uniform")
    
    weights = optimizer.optimize(X, y)
    
    assert np.allclose(weights, [1/3, 1/3, 1/3])


def test_performance_weights(mock_models, sample_data):
    """Test performance-based weight calculation."""
    X, y = sample_data
    optimizer = WeightOptimizer(mock_models, method="performance")
    
    weights = optimizer.optimize(X, y)
    
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert all(w >= 0 for w in weights)


@patch("optuna.create_study")
def test_optuna_weights(mock_create_study, mock_models, sample_data):
    """Test Optuna weight optimization."""
    X, y = sample_data
    
    # Mock Optuna study
    mock_study = Mock()
    mock_study.best_params = {"weight_0": 0.5, "weight_1": 0.3}
    mock_study.best_value = 0.85
    mock_create_study.return_value = mock_study
    
    optimizer = WeightOptimizer(mock_models, method="optuna", n_trials=10)
    result = optimizer.optimize_with_details(X, y)
    
    assert "weights" in result
    assert len(result["weights"]) == 3
    assert np.isclose(np.sum(result["weights"]), 1.0)


def test_grid_weights_two_models(sample_data):
    """Test grid weight optimization with 2 models."""
    X, y = sample_data
    
    models = []
    for i in range(2):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y + np.random.randn(100) * 0.3)
        models.append(model)
    
    optimizer = WeightOptimizer(models, method="grid")
    result = optimizer.optimize_with_details(X, y)
    
    assert len(result["weights"]) == 2
    assert np.isclose(np.sum(result["weights"]), 1.0)


def test_grid_weights_three_models(sample_data):
    """Test grid weight optimization with 3 models."""
    X, y = sample_data
    
    models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y + np.random.randn(100) * 0.3)
        models.append(model)
    
    optimizer = WeightOptimizer(models, method="grid")
    result = optimizer.optimize_with_details(X, y)
    
    assert len(result["weights"]) == 3
    assert np.isclose(np.sum(result["weights"]), 1.0)


@patch("optuna.create_study")
def test_grid_weights_many_models_fallback(
    mock_create_study, sample_data
):
    """Test grid weights falls back to Optuna for many models."""
    X, y = sample_data
    
    # Create 5 models
    models = []
    for i in range(5):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y + np.random.randn(100) * 0.3)
        models.append(model)
    
    # Mock Optuna
    mock_study = Mock()
    mock_study.best_params = {
        "weight_0": 0.3,
        "weight_1": 0.25,
        "weight_2": 0.2,
        "weight_3": 0.15,
    }
    mock_study.best_value = 0.8
    mock_create_study.return_value = mock_study
    
    optimizer = WeightOptimizer(models, method="grid")
    result = optimizer.optimize_with_details(X, y)
    
    # Should fall back to Optuna
    assert len(result["weights"]) == 5


def test_unknown_method(mock_models, sample_data):
    """Test unknown optimization method."""
    X, y = sample_data
    optimizer = WeightOptimizer(mock_models, method="invalid")
    
    with pytest.raises(ValueError, match="Unknown optimization method"):
        optimizer.optimize(X, y)


def test_optimize_with_details_uniform(mock_models, sample_data):
    """Test optimize_with_details with uniform method."""
    X, y = sample_data
    optimizer = WeightOptimizer(mock_models, method="uniform")
    
    result = optimizer.optimize_with_details(X, y)
    
    assert "weights" in result
    assert "method" in result
    assert result["method"] == "uniform"


def test_optimize_with_details_performance(mock_models, sample_data):
    """Test optimize_with_details with performance method."""
    X, y = sample_data
    optimizer = WeightOptimizer(mock_models, method="performance")
    
    result = optimizer.optimize_with_details(X, y)
    
    assert "weights" in result
    assert "method" in result
    assert result["method"] == "performance"


@patch("optuna.create_study")
def test_optuna_objective_constraint(mock_create_study, mock_models, sample_data):
    """Test Optuna objective enforces weight constraint."""
    X, y = sample_data
    
    optimizer = WeightOptimizer(mock_models, method="optuna", n_trials=5)
    optimizer.X = X
    optimizer.y = y
    
    # Create mock trial with invalid weights (sum > 1)
    mock_trial = Mock()
    mock_trial.suggest_float = Mock(side_effect=[0.6, 0.6])  # Sum = 1.2 > 1
    
    score = optimizer._optuna_objective(mock_trial)
    
    # Should return very low score for invalid weights
    assert score == -999.0