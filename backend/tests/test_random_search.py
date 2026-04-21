"""Tests for random search optimizer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.random_search import RandomSearchOptimizer


@pytest.fixture
def model():
    """Create test model."""
    return RandomForestRegressor(n_estimators=10, random_state=42)


@pytest.fixture
def param_distributions():
    """Create test parameter distributions."""
    return {
        "max_depth": randint(5, 15),
        "min_samples_split": uniform(0.01, 0.2),
    }


@pytest.fixture
def optimizer(model, param_distributions):
    """Create random search optimizer."""
    return RandomSearchOptimizer(
        model, param_distributions, n_iter=10, cv=3, random_state=42
    )


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_random_search_initialization(model, param_distributions):
    """Test random search initialization."""
    optimizer = RandomSearchOptimizer(
        model, param_distributions, n_iter=50, cv=5
    )
    assert optimizer.param_distributions == param_distributions
    assert optimizer.n_iter == 50
    assert optimizer.cv == 5


def test_random_search_empty_distributions(model):
    """Test random search rejects empty distributions."""
    with pytest.raises(ValueError, match="Parameter distributions cannot be empty"):
        RandomSearchOptimizer(model, {})


def test_random_search_invalid_n_iter(model, param_distributions):
    """Test random search rejects invalid n_iter."""
    with pytest.raises(ValueError, match="n_iter must be >= 1"):
        RandomSearchOptimizer(model, param_distributions, n_iter=0)


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_calls_sklearn(mock_random_search, optimizer, sample_data):
    """Test random search calls sklearn RandomizedSearchCV."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10, "min_samples_split": 0.05}
    mock_instance.best_score_ = 0.72
    mock_instance.cv_results_ = {"params": [{}], "mean_test_score": [0.72]}
    mock_random_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert mock_random_search.called
    assert results["best_score"] == 0.72


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_returns_correct_structure(mock_random_search, optimizer, sample_data):
    """Test random search returns correct result structure."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 8}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_random_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert "best_params" in results
    assert "best_score" in results
    assert "cv_results" in results
    assert "total_trials" in results
    assert "random_state" in results


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_stores_results(mock_random_search, optimizer, sample_data):
    """Test random search stores results internally."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 12}
    mock_instance.best_score_ = 0.78
    mock_instance.cv_results_ = {"params": [{}]}
    mock_random_search.return_value = mock_instance
    
    optimizer.search(X, y)
    
    assert optimizer.get_best_params() == {"max_depth": 12}
    assert optimizer.get_best_score() == 0.78


def test_random_search_validates_inputs(optimizer):
    """Test random search validates inputs."""
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError, match="Empty input"):
        optimizer.search(X_empty, y_empty)


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_handles_exceptions(mock_random_search, optimizer, sample_data):
    """Test random search handles exceptions."""
    X, y = sample_data
    
    # Setup mock to raise exception
    mock_random_search.return_value.fit.side_effect = RuntimeError("Test error")
    
    with pytest.raises(RuntimeError):
        optimizer.search(X, y)


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_n_iter_respected(mock_random_search, sample_data):
    """Test random search respects n_iter parameter."""
    X, y = sample_data
    model = RandomForestRegressor(random_state=42)
    param_distributions = {"max_depth": randint(5, 15)}
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.70
    mock_instance.cv_results_ = {"params": [{}]}
    mock_random_search.return_value = mock_instance
    
    optimizer = RandomSearchOptimizer(model, param_distributions, n_iter=25, cv=3)
    optimizer.search(X, y)
    
    # Verify n_iter was passed
    call_kwargs = mock_random_search.call_args[1]
    assert call_kwargs["n_iter"] == 25


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_random_state_set(mock_random_search, optimizer, sample_data):
    """Test random search uses random state."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_random_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert results["random_state"] == 42


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_random_search_with_different_seeds(mock_random_search, sample_data):
    """Test random search with different seeds."""
    X, y = sample_data
    model = RandomForestRegressor(random_state=42)
    param_distributions = {"max_depth": randint(5, 15)}
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_random_search.return_value = mock_instance
    
    optimizer1 = RandomSearchOptimizer(
        model, param_distributions, n_iter=10, random_state=42
    )
    optimizer2 = RandomSearchOptimizer(
        model, param_distributions, n_iter=10, random_state=100
    )
    
    results1 = optimizer1.search(X, y)
    results2 = optimizer2.search(X, y)
    
    # Different random states should be recorded
    assert results1["random_state"] != results2["random_state"]