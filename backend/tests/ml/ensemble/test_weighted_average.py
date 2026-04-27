"""Tests for weighted average ensemble."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.weighted_average import WeightedAverageEnsemble


@pytest.fixture
def mock_models():
    """Create mock models."""
    models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.ones(100) * (i + 1))
        models.append(model)
    return models


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_weighted_average_initialization(mock_models):
    """Test weighted average initialization."""
    ensemble = WeightedAverageEnsemble(mock_models, weight_method="uniform")
    assert len(ensemble.base_models) == 3
    assert ensemble.weight_method == "uniform"


def test_weighted_average_empty_models():
    """Test weighted average with empty models."""
    with pytest.raises(ValueError, match="cannot be empty"):
        WeightedAverageEnsemble([])


def test_uniform_weights(mock_models, sample_data):
    """Test uniform weight calculation."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models, weight_method="uniform")
    ensemble.fit(X, y)
    
    assert np.allclose(ensemble.weights, [1/3, 1/3, 1/3])


def test_performance_weights(mock_models, sample_data):
    """Test performance-based weight calculation."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models, weight_method="performance")
    ensemble.fit(X, y)
    
    assert ensemble.weights is not None
    assert len(ensemble.weights) == 3
    assert np.isclose(np.sum(ensemble.weights), 1.0)


def test_custom_weights(mock_models, sample_data):
    """Test custom weights."""
    X, y = sample_data
    weights = np.array([0.5, 0.3, 0.2])
    
    ensemble = WeightedAverageEnsemble(mock_models, weights=weights)
    ensemble.fit(X, y)
    
    assert np.allclose(ensemble.weights, weights)


def test_weighted_average_predict(mock_models, sample_data):
    """Test weighted average predict."""
    X, y = sample_data
    
    weights = np.array([0.5, 0.3, 0.2])
    ensemble = WeightedAverageEnsemble(mock_models, weights=weights)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    
    # Expected: 0.5*1 + 0.3*2 + 0.2*3 = 1.7
    assert predictions.shape == (100,)
    assert np.allclose(predictions, 1.7)


def test_predict_before_fit_raises_error(mock_models, sample_data):
    """Test predict before fit raises error."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models)
    
    with pytest.raises(ValueError, match="must be fitted"):
        ensemble.predict(X)


def test_set_weights(mock_models, sample_data):
    """Test set_weights method."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models)
    ensemble.fit(X, y)
    
    new_weights = np.array([0.6, 0.3, 0.1])
    ensemble.set_weights(new_weights)
    
    assert np.allclose(ensemble.weights, new_weights)


def test_set_weights_invalid_sum(mock_models, sample_data):
    """Test set_weights with invalid sum."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models)
    ensemble.fit(X, y)
    
    invalid_weights = np.array([0.5, 0.3, 0.3])  # Sum = 1.1
    
    with pytest.raises(ValueError, match="sum to 1.0"):
        ensemble.set_weights(invalid_weights)


def test_set_weights_negative(mock_models, sample_data):
    """Test set_weights with negative values."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models)
    ensemble.fit(X, y)
    
    invalid_weights = np.array([0.6, 0.5, -0.1])
    
    with pytest.raises(ValueError, match="non-negative"):
        ensemble.set_weights(invalid_weights)


def test_weighted_average_repr(mock_models):
    """Test string representation."""
    ensemble = WeightedAverageEnsemble(mock_models, weight_method="performance")
    repr_str = repr(ensemble)
    
    assert "WeightedAverageEnsemble" in repr_str
    assert "n_models=3" in repr_str
    assert "performance" in repr_str


def test_optimized_method_no_weights(mock_models, sample_data):
    """Test optimized method with no weights set."""
    X, y = sample_data
    ensemble = WeightedAverageEnsemble(mock_models, weight_method="optimized")
    ensemble.fit(X, y)
    
    # Should fall back to uniform weights
    assert ensemble.weights is not None


def test_weighted_average_save_and_load(mock_models, sample_data, tmp_path):
    """Test save and load weighted average ensemble."""
    X, y = sample_data
    
    weights = np.array([0.5, 0.3, 0.2])
    ensemble = WeightedAverageEnsemble(mock_models, weights=weights)
    ensemble.fit(X, y)
    
    save_path = tmp_path / "weighted.joblib"
    ensemble.save(save_path)
    
    loaded = WeightedAverageEnsemble.load(save_path)
    assert np.allclose(loaded.weights, weights)