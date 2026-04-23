"""Tests for Optuna optimizer."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.optuna_optimizer import OptunaOptimizer


@pytest.fixture
def model():
    """Create test model."""
    return RandomForestRegressor(n_estimators=10, random_state=42)


@pytest.fixture
def search_space():
    """Create test search space."""
    return {
        "max_depth": {"type": "int", "low": 5, "high": 15},
        "min_samples_split": {"type": "float", "low": 0.01, "high": 0.2},
        "n_estimators": {"type": "int", "low": 10, "high": 50, "step": 10},
    }


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_optuna_optimizer_initialization(model, search_space):
    """Test Optuna optimizer initialization."""
    optimizer = OptunaOptimizer(
        model=model,
        search_space=search_space,
        n_trials=10,
        direction="maximize",
    )
    assert optimizer.n_trials == 10
    assert optimizer.direction == "maximize"
    assert optimizer.search_space == search_space


def test_invalid_direction(model, search_space):
    """Test invalid direction raises error."""
    with pytest.raises(ValueError, match="Invalid direction"):
        OptunaOptimizer(
            model=model,
            search_space=search_space,
            direction="invalid",
        )


@patch("optuna.create_study")
def test_optuna_search_creates_study(mock_create_study, model, search_space, sample_data):
    """Test that search creates Optuna study."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {"max_depth": 10}
    mock_study.best_value = 0.78
    mock_study.best_trial.number = 5
    mock_study.trials = [Mock() for _ in range(10)]
    mock_create_study.return_value = mock_study
    
    optimizer = OptunaOptimizer(
        model=model,
        search_space=search_space,
        n_trials=10,
    )
    
    results = optimizer.search(X, y)
    
    assert mock_create_study.called
    assert results["best_score"] == 0.78
    assert results["n_trials"] == 10


@patch("optuna.create_study")
def test_optuna_search_returns_correct_structure(
    mock_create_study, model, search_space, sample_data
):
    """Test search returns correct result structure."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {"max_depth": 10}
    mock_study.best_value = 0.80
    mock_study.best_trial.number = 8
    mock_study.trials = [
        Mock(state=MagicMock(name="COMPLETE")) for _ in range(8)
    ] + [Mock(state=MagicMock(name="PRUNED")) for _ in range(2)]
    mock_create_study.return_value = mock_study
    
    optimizer = OptunaOptimizer(model, search_space, n_trials=10)
    results = optimizer.search(X, y)
    
    assert "best_params" in results
    assert "best_score" in results
    assert "best_trial" in results
    assert "n_trials" in results
    assert "n_complete" in results
    assert "n_pruned" in results
    assert "study" in results


@patch("optuna.create_study")
def test_optuna_search_with_sampler(mock_create_study, model, search_space, sample_data):
    """Test search with custom sampler."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {}
    mock_study.best_value = 0.75
    mock_study.best_trial.number = 0
    mock_study.trials = [Mock()]
    mock_create_study.return_value = mock_study
    
    from optuna.samplers import RandomSampler
    
    sampler = RandomSampler(seed=42)
    optimizer = OptunaOptimizer(
        model, search_space, n_trials=5, sampler=sampler
    )
    optimizer.search(X, y)
    
    # Verify sampler was passed
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["sampler"] == sampler


@patch("optuna.create_study")
def test_optuna_search_with_pruner(mock_create_study, model, search_space, sample_data):
    """Test search with custom pruner."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {}
    mock_study.best_value = 0.75
    mock_study.best_trial.number = 0
    mock_study.trials = [Mock()]
    mock_create_study.return_value = mock_study
    
    from optuna.pruners import MedianPruner
    
    pruner = MedianPruner()
    optimizer = OptunaOptimizer(
        model, search_space, n_trials=5, pruner=pruner
    )
    optimizer.search(X, y)
    
    # Verify pruner was passed
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["pruner"] == pruner


def test_suggest_params_float(model, search_space):
    """Test parameter suggestion for float type."""
    optimizer = OptunaOptimizer(model, search_space, n_trials=1)
    
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 0.1
    mock_trial.suggest_int.return_value = 10
    
    params = optimizer._suggest_params(mock_trial)
    
    assert "min_samples_split" in params
    assert mock_trial.suggest_float.called


def test_suggest_params_int(model, search_space):
    """Test parameter suggestion for int type."""
    optimizer = OptunaOptimizer(model, search_space, n_trials=1)
    
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 10
    mock_trial.suggest_float.return_value = 0.1
    
    params = optimizer._suggest_params(mock_trial)
    
    assert "max_depth" in params
    assert mock_trial.suggest_int.called


def test_suggest_params_categorical():
    """Test parameter suggestion for categorical type."""
    model = RandomForestRegressor()
    search_space = {
        "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}
    }
    
    optimizer = OptunaOptimizer(model, search_space, n_trials=1)
    
    mock_trial = MagicMock()
    mock_trial.suggest_categorical.return_value = "gini"
    
    params = optimizer._suggest_params(mock_trial)
    
    assert "criterion" in params
    assert mock_trial.suggest_categorical.called


def test_suggest_params_unknown_type(model):
    """Test unknown parameter type raises error."""
    search_space = {"param": {"type": "unknown"}}
    optimizer = OptunaOptimizer(model, search_space, n_trials=1)
    
    mock_trial = MagicMock()
    
    with pytest.raises(ValueError, match="Unknown parameter type"):
        optimizer._suggest_params(mock_trial)


def test_search_validates_inputs(model, search_space):
    """Test search validates inputs."""
    optimizer = OptunaOptimizer(model, search_space, n_trials=1)
    
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError, match="Empty input"):
        optimizer.search(X_empty, y_empty)


@patch("optuna.create_study")
def test_search_stores_results(mock_create_study, model, search_space, sample_data):
    """Test search stores results internally."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {"max_depth": 12}
    mock_study.best_value = 0.82
    mock_study.best_trial.number = 3
    mock_study.trials = [Mock()]
    mock_create_study.return_value = mock_study
    
    optimizer = OptunaOptimizer(model, search_space, n_trials=5)
    optimizer.search(X, y)
    
    assert optimizer.get_best_params() == {"max_depth": 12}
    assert optimizer.get_best_score() == 0.82


@patch("optuna.create_study")
def test_search_with_storage(mock_create_study, model, search_space, sample_data):
    """Test search with storage URL."""
    X, y = sample_data
    
    # Setup mock
    mock_study = MagicMock()
    mock_study.best_params = {}
    mock_study.best_value = 0.75
    mock_study.best_trial.number = 0
    mock_study.trials = [Mock()]
    mock_create_study.return_value = mock_study
    
    storage = "sqlite:///test.db"
    optimizer = OptunaOptimizer(
        model, search_space, n_trials=5, storage=storage
    )
    optimizer.search(X, y)
    
    # Verify storage was passed
    call_kwargs = mock_create_study.call_args[1]
    assert call_kwargs["storage"] == storage


@patch("optuna.create_study")
def test_search_handles_exceptions(mock_create_study, model, search_space, sample_data):
    """Test search handles exceptions."""
    X, y = sample_data
    
    # Setup mock to raise exception
    mock_create_study.side_effect = RuntimeError("Test error")
    
    optimizer = OptunaOptimizer(model, search_space, n_trials=5)
    
    with pytest.raises(RuntimeError):
        optimizer.search(X, y)