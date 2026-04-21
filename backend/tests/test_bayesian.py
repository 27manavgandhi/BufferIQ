"""Tests for Bayesian optimizer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.bayesian import SKOPT_AVAILABLE, BayesianOptimizer

# Skip all tests if skopt not available
pytestmark = pytest.mark.skipif(
    not SKOPT_AVAILABLE,
    reason="scikit-optimize not installed"
)


@pytest.fixture
def model():
    """Create test model."""
    return RandomForestRegressor(n_estimators=10, random_state=42)


@pytest.fixture
def search_spaces():
    """Create test search spaces."""
    if SKOPT_AVAILABLE:
        from skopt.space import Integer, Real
        return {
            "max_depth": Integer(5, 15),
            "min_samples_split": Real(0.01, 0.2),
        }
    return {}


@pytest.fixture
def optimizer(model, search_spaces):
    """Create Bayesian optimizer."""
    return BayesianOptimizer(
        model, search_spaces, n_iter=10, cv=3, random_state=42
    )


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_bayesian_initialization(model, search_spaces):
    """Test Bayesian optimizer initialization."""
    optimizer = BayesianOptimizer(
        model, search_spaces, n_iter=50, cv=5
    )
    assert optimizer.search_spaces == search_spaces
    assert optimizer.n_iter == 50
    assert optimizer.cv == 5


def test_bayesian_empty_search_spaces(model):
    """Test Bayesian optimizer rejects empty search spaces."""
    with pytest.raises(ValueError, match="Search spaces cannot be empty"):
        BayesianOptimizer(model, {})


def test_bayesian_invalid_n_iter(model, search_spaces):
    """Test Bayesian optimizer rejects invalid n_iter."""
    with pytest.raises(ValueError, match="n_iter must be >= 1"):
        BayesianOptimizer(model, search_spaces, n_iter=0)


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_calls_skopt(mock_bayes_search, optimizer, sample_data):
    """Test Bayesian optimizer calls skopt BayesSearchCV."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 10, "min_samples_split": 0.05}
    mock_instance.best_score_ = 0.76
    mock_instance.cv_results_ = {"params": [{}], "mean_test_score": [0.76]}
    mock_bayes_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert mock_bayes_search.called
    assert results["best_score"] == 0.76


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_returns_correct_structure(mock_bayes_search, optimizer, sample_data):
    """Test Bayesian optimizer returns correct result structure."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 8}
    mock_instance.best_score_ = 0.78
    mock_instance.cv_results_ = {"params": [{}]}
    mock_bayes_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert "best_params" in results
    assert "best_score" in results
    assert "cv_results" in results
    assert "total_trials" in results
    assert "random_state" in results


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_stores_results(mock_bayes_search, optimizer, sample_data):
    """Test Bayesian optimizer stores results internally."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {"max_depth": 12}
    mock_instance.best_score_ = 0.79
    mock_instance.cv_results_ = {"params": [{}]}
    mock_bayes_search.return_value = mock_instance
    
    optimizer.search(X, y)
    
    assert optimizer.get_best_params() == {"max_depth": 12}
    assert optimizer.get_best_score() == 0.79


def test_bayesian_validates_inputs(optimizer):
    """Test Bayesian optimizer validates inputs."""
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError, match="Empty input"):
        optimizer.search(X_empty, y_empty)


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_handles_exceptions(mock_bayes_search, optimizer, sample_data):
    """Test Bayesian optimizer handles exceptions."""
    X, y = sample_data
    
    # Setup mock to raise exception
    mock_bayes_search.return_value.fit.side_effect = RuntimeError("Test error")
    
    with pytest.raises(RuntimeError):
        optimizer.search(X, y)


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_n_iter_respected(mock_bayes_search, sample_data, search_spaces):
    """Test Bayesian optimizer respects n_iter parameter."""
    X, y = sample_data
    model = RandomForestRegressor(random_state=42)
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.70
    mock_instance.cv_results_ = {"params": [{}]}
    mock_bayes_search.return_value = mock_instance
    
    optimizer = BayesianOptimizer(model, search_spaces, n_iter=30, cv=3)
    optimizer.search(X, y)
    
    # Verify n_iter was passed
    call_kwargs = mock_bayes_search.call_args[1]
    assert call_kwargs["n_iter"] == 30


@patch("bufferiq.ml.optimization.bayesian.BayesSearchCV")
def test_bayesian_random_state_set(mock_bayes_search, optimizer, sample_data):
    """Test Bayesian optimizer uses random state."""
    X, y = sample_data
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.best_params_ = {}
    mock_instance.best_score_ = 0.75
    mock_instance.cv_results_ = {"params": [{}]}
    mock_bayes_search.return_value = mock_instance
    
    results = optimizer.search(X, y)
    
    assert results["random_state"] == 42