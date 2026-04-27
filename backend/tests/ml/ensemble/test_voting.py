"""Tests for voting ensemble."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.voting import VotingEnsemble


@pytest.fixture
def mock_models():
    """Create mock models."""
    models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        # Each model returns different predictions
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


def test_voting_ensemble_initialization(mock_models):
    """Test voting ensemble initialization."""
    ensemble = VotingEnsemble(mock_models)
    assert len(ensemble.base_models) == 3
    assert np.allclose(ensemble.weights, [1/3, 1/3, 1/3])


def test_voting_ensemble_custom_weights(mock_models):
    """Test voting ensemble with custom weights."""
    weights = np.array([0.5, 0.3, 0.2])
    ensemble = VotingEnsemble(mock_models, weights=weights)
    assert np.allclose(ensemble.weights, weights)


def test_voting_ensemble_empty_models():
    """Test voting ensemble with empty models list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        VotingEnsemble([])


def test_weights_must_sum_to_one(mock_models):
    """Test weights validation - must sum to 1."""
    weights = np.array([0.5, 0.3, 0.3])  # Sum = 1.1
    
    with pytest.raises(ValueError, match="sum to 1.0"):
        VotingEnsemble(mock_models, weights=weights)


def test_weights_must_be_non_negative(mock_models):
    """Test weights validation - must be non-negative."""
    weights = np.array([0.6, 0.5, -0.1])
    
    with pytest.raises(ValueError, match="non-negative"):
        VotingEnsemble(mock_models, weights=weights)


def test_weights_length_mismatch(mock_models):
    """Test weights validation - length must match models."""
    weights = np.array([0.5, 0.5])  # Only 2 weights for 3 models
    
    with pytest.raises(ValueError, match="must match"):
        VotingEnsemble(mock_models, weights=weights)


def test_voting_ensemble_fit(mock_models, sample_data):
    """Test voting ensemble fit."""
    X, y = sample_data
    ensemble = VotingEnsemble(mock_models)
    
    result = ensemble.fit(X, y)
    
    assert result is ensemble
    assert ensemble._is_fitted is True


def test_voting_ensemble_predict(mock_models, sample_data):
    """Test voting ensemble predict."""
    X, y = sample_data
    ensemble = VotingEnsemble(mock_models)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    
    assert predictions.shape == (100,)
    assert all(model.predict.called for model in mock_models)


def test_voting_ensemble_predict_weighted(mock_models, sample_data):
    """Test weighted voting predictions."""
    X, y = sample_data
    
    # Set specific predictions for testing
    mock_models[0].predict.return_value = np.ones(100)
    mock_models[1].predict.return_value = np.ones(100) * 2
    mock_models[2].predict.return_value = np.ones(100) * 3
    
    weights = np.array([0.5, 0.3, 0.2])
    ensemble = VotingEnsemble(mock_models, weights=weights)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    
    # Expected: 0.5*1 + 0.3*2 + 0.2*3 = 1.7
    assert np.allclose(predictions, 1.7)


def test_predict_before_fit_raises_error(mock_models, sample_data):
    """Test predict before fit raises error."""
    X, y = sample_data
    ensemble = VotingEnsemble(mock_models)
    
    with pytest.raises(ValueError, match="must be fitted"):
        ensemble.predict(X)


def test_voting_ensemble_repr(mock_models):
    """Test string representation."""
    ensemble = VotingEnsemble(mock_models)
    repr_str = repr(ensemble)
    
    assert "VotingEnsemble" in repr_str
    assert "n_models=3" in repr_str
    assert "soft" in repr_str


def test_voting_with_single_model():
    """Test voting with single model."""
    model = Mock(spec=BaseEstimator)
    model.predict = Mock(return_value=np.ones(100))
    
    ensemble = VotingEnsemble([model], weights=np.array([1.0]))
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    
    assert np.allclose(predictions, 1.0)


def test_voting_save_and_load(mock_models, sample_data, tmp_path):
    """Test save and load voting ensemble."""
    X, y = sample_data
    
    ensemble = VotingEnsemble(mock_models, weights=np.array([0.5, 0.3, 0.2]))
    ensemble.fit(X, y)
    
    save_path = tmp_path / "voting.joblib"
    ensemble.save(save_path)
    
    loaded = VotingEnsemble.load(save_path)
    assert np.allclose(loaded.weights, [0.5, 0.3, 0.2])