"""Tests for base optimizer class."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.base import BaseOptimizer


class DummyOptimizer(BaseOptimizer):
    """Dummy optimizer for testing abstract base class."""

    def search(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Dummy search implementation."""
        self._best_params = {"dummy_param": 1}
        self._best_score = 0.75
        self._search_results = {
            "best_params": self._best_params,
            "best_score": self._best_score,
        }
        return self._search_results


@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""
    return RandomForestRegressor(random_state=42)


@pytest.fixture
def optimizer(dummy_model):
    """Create dummy optimizer instance."""
    return DummyOptimizer(dummy_model, cv=5, scoring="r2")


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_optimizer_initialization(dummy_model):
    """Test optimizer initialization."""
    optimizer = DummyOptimizer(dummy_model, cv=5, scoring="r2")
    assert optimizer.model == dummy_model
    assert optimizer.cv == 5
    assert optimizer.scoring == "r2"
    assert optimizer.random_state == 42


def test_optimizer_invalid_cv(dummy_model):
    """Test optimizer rejects invalid cv."""
    with pytest.raises(ValueError, match="cv must be >= 2"):
        DummyOptimizer(dummy_model, cv=1)


def test_validate_inputs_empty_arrays(optimizer):
    """Test validation rejects empty arrays."""
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError, match="Empty input arrays"):
        optimizer.validate_inputs(X_empty, y_empty)


def test_validate_inputs_mismatched_shapes(optimizer):
    """Test validation rejects mismatched shapes."""
    X = np.random.randn(100, 10)
    y = np.random.randn(50)
    
    with pytest.raises(ValueError, match="same length"):
        optimizer.validate_inputs(X, y)


def test_validate_inputs_nan_in_X(optimizer):
    """Test validation rejects NaN in X."""
    X = np.random.randn(100, 10)
    X[0, 0] = np.nan
    y = np.random.randn(100)
    
    with pytest.raises(ValueError, match="X contains NaN"):
        optimizer.validate_inputs(X, y)


def test_validate_inputs_nan_in_y(optimizer):
    """Test validation rejects NaN in y."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    y[0] = np.nan
    
    with pytest.raises(ValueError, match="y contains NaN"):
        optimizer.validate_inputs(X, y)


def test_validate_inputs_inf_in_X(optimizer):
    """Test validation rejects inf in X."""
    X = np.random.randn(100, 10)
    X[0, 0] = np.inf
    y = np.random.randn(100)
    
    with pytest.raises(ValueError, match="X contains infinite"):
        optimizer.validate_inputs(X, y)


def test_validate_inputs_inf_in_y(optimizer):
    """Test validation rejects inf in y."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    y[0] = np.inf
    
    with pytest.raises(ValueError, match="y contains infinite"):
        optimizer.validate_inputs(X, y)


def test_validate_inputs_valid_data(optimizer, sample_data):
    """Test validation passes for valid data."""
    X, y = sample_data
    optimizer.validate_inputs(X, y)  # Should not raise


def test_get_best_params_before_search(optimizer):
    """Test get_best_params returns None before search."""
    assert optimizer.get_best_params() is None


def test_get_best_params_after_search(optimizer, sample_data):
    """Test get_best_params returns params after search."""
    X, y = sample_data
    optimizer.search(X, y)
    
    params = optimizer.get_best_params()
    assert params is not None
    assert "dummy_param" in params


def test_get_best_score_before_search(optimizer):
    """Test get_best_score returns None before search."""
    assert optimizer.get_best_score() is None


def test_get_best_score_after_search(optimizer, sample_data):
    """Test get_best_score returns score after search."""
    X, y = sample_data
    optimizer.search(X, y)
    
    score = optimizer.get_best_score()
    assert score is not None
    assert isinstance(score, float)


def test_get_search_results_before_search(optimizer):
    """Test get_search_results returns None before search."""
    assert optimizer.get_search_results() is None


def test_get_search_results_after_search(optimizer, sample_data):
    """Test get_search_results returns results after search."""
    X, y = sample_data
    results = optimizer.search(X, y)
    
    stored_results = optimizer.get_search_results()
    assert stored_results is not None
    assert stored_results == results