"""Tests for grid search optimizer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.grid_search import GridSearchOptimizer


@pytest.fixture
def model():
    """Create test model."""
    return RandomForestRegressor(n_estimators=10, random_state=42)


@pytest.fixture
def param_grid():
    """Create test parameter grid."""
    return {
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    }


@pytest.fixture
def optimizer(model, param_grid):
    """Create grid search optimizer."""
    return GridSearchOptimizer(model, param_grid, cv=3, random_state=42)


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_grid_search_initialization(model, param_grid):
    """Test grid search initialization."""
    optimizer = GridSearchOptimizer(model, param_grid, cv=5)
    assert optimizer.param_grid == param_grid
    assert optimizer.total_combinations == 4  # 2 * 2


def test_grid_search_empty_param_grid(model):
    """Test grid search rejects empty param grid."""
    with pytest.raises(ValueError, match="Parameter grid cannot be empty"):
        GridSearchOptimizer(model, {})


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_calls_sklearn(mock_grid_search, optimizer, sample_data):
    """Test grid search calls sklearn GridSearchCV."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10, "min_samples_split": 2}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}], "mean_test_score": [0.75]}
    mock_grid_search.return_value = mock_instance
    
    # Run search
    results = optimizer.search(X, y)
    
    # Verify GridSearchCV was called
    assert mock_grid_search.called
    assert results["best_score"] == 0.75


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_returns_correct_structure(mock_grid_search, optimizer, sample_data):
    """Test grid search returns correct result structure."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10}
    mock_instance.best_score_ = 0.80
    mock_instance.cv_results_ = {
        "params": [{"max_depth": 5}, {"max_depth": 10}],
        "mean_test_score": [0.70, 0.80],
    }
    mock_grid_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert "best_params" in results
    assert "best_score" in results
    assert "cv_results" in results
    assert "total_trials" in results
    assert "random_state" in results


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_stores_results(mock_grid_search, optimizer, sample_data):
    """Test grid search stores results internally."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_grid_search.return_value = mock_instance
    
    optimizer.search(X, y)
    
    assert optimizer.get_best_params() == {"max_depth": 10}
    assert optimizer.get_best_score() == 0.75


def test_grid_search_validates_inputs(optimizer):
    """Test grid search validates inputs."""
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError, match="Empty input"):
        optimizer.search(X_empty, y_empty)


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_handles_exceptions(mock_grid_search, optimizer, sample_data):
    """Test grid search handles exceptions."""
    X, y = sample_data
    
    # Setup mock to raise exception
    mock_grid_search.return_value.fit.side_effect = RuntimeError("Test error")
    
    with pytest.raises(RuntimeError):
        optimizer.search(X, y)


def test_grid_search_total_combinations():
    """Test total combinations calculation."""
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
    }
    optimizer = GridSearchOptimizer(model, param_grid)
    
    assert optimizer.total_combinations == 18  # 3 * 2 * 3


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_with_single_param(mock_grid_search, sample_data):
    """Test grid search with single parameter."""
    X, y = sample_data
    model = RandomForestRegressor(random_state=42)
    param_grid = {"max_depth": [5, 10, 15]}
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10}
    mock_instance.best_score_ = 0.70
    mock_instance.cv_results_ = {"params": [{}]}
    mock_grid_search.return_value = mock_instance
    
    optimizer = GridSearchOptimizer(model, param_grid, cv=3)
    results = optimizer.search(X, y)
    
    assert results["best_params"]["max_depth"] == 10
    assert optimizer.total_combinations == 3


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_random_state_set(mock_grid_search, optimizer, sample_data):
    """Test grid search uses random state."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_grid_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert results["random_state"] == 42


@patch("sklearn.model_selection.GridSearchCV")
def test_grid_search_with_cv_folds(mock_grid_search, sample_data):
    """Test grid search respects cv_folds."""
    X, y = sample_data
    model = RandomForestRegressor(random_state=42)
    param_grid = {"max_depth": [5, 10]}
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_grid_search.return_value = mock_instance
    
    optimizer = GridSearchOptimizer(model, param_grid, cv=7)
    optimizer.search(X, y)
    
    # Verify cv parameter was passed
    call_kwargs = mock_grid_search.call_args[1]
    assert call_kwargs["cv"] == 7