"""Tests for stacking ensemble."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from bufferiq.ml.ensemble.stacking import StackingEnsemble


@pytest.fixture
def mock_models():
    """Create mock models."""
    models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(100))
        model.fit = Mock(return_value=None)
        models.append(model)
    return models


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_stacking_ensemble_initialization(mock_models):
    """Test stacking ensemble initialization."""
    meta_learner = Ridge()
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    
    assert len(ensemble.base_models) == 3
    assert ensemble.cv == 5
    assert ensemble.passthrough is False


def test_stacking_ensemble_empty_models():
    """Test stacking with empty models list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        StackingEnsemble([], Ridge())


def test_stacking_ensemble_invalid_cv(mock_models):
    """Test stacking with invalid cv."""
    with pytest.raises(ValueError, match="cv must be >= 2"):
        StackingEnsemble(mock_models, Ridge(), cv=1)


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_ensemble_fit(mock_cross_val_predict, mock_models, sample_data):
    """Test stacking ensemble fit."""
    X, y = sample_data
    
    # Mock out-of-fold predictions
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    result = ensemble.fit(X, y)
    
    assert result is ensemble
    assert ensemble._is_fitted is True
    assert mock_cross_val_predict.call_count == 3
    assert meta_learner.fit.called


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_ensemble_fit_with_passthrough(
    mock_cross_val_predict, mock_models, sample_data
):
    """Test stacking with passthrough=True."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5, passthrough=True)
    ensemble.fit(X, y)
    
    # Check that meta_learner.fit was called with features including original X
    call_args = meta_learner.fit.call_args[0]
    meta_features = call_args[0]
    
    # Meta features should have shape (100, 3 + 10) = (100, 13)
    # 3 from base models + 10 from original features
    assert meta_features.shape[1] == 13


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_ensemble_predict(mock_cross_val_predict, mock_models, sample_data):
    """Test stacking ensemble predict."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    meta_learner.predict = Mock(return_value=np.random.randn(100))
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    
    assert predictions.shape == (100,)
    assert meta_learner.predict.called


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_ensemble_predict_with_passthrough(
    mock_cross_val_predict, mock_models, sample_data
):
    """Test stacking predict with passthrough."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    meta_learner.predict = Mock(return_value=np.random.randn(100))
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5, passthrough=True)
    ensemble.fit(X, y)
    ensemble.predict(X)
    
    # Check that meta_learner.predict was called with features including original X
    call_args = meta_learner.predict.call_args[0]
    meta_input = call_args[0]
    
    # Should have shape (100, 13)
    assert meta_input.shape[1] == 13


def test_predict_before_fit_raises_error(mock_models, sample_data):
    """Test predict before fit raises error."""
    X, y = sample_data
    ensemble = StackingEnsemble(mock_models, Ridge())
    
    with pytest.raises(ValueError, match="must be fitted"):
        ensemble.predict(X)


def test_stacking_ensemble_repr(mock_models):
    """Test string representation."""
    meta_learner = Ridge()
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    repr_str = repr(ensemble)
    
    assert "StackingEnsemble" in repr_str
    assert "n_base_models=3" in repr_str
    assert "Ridge" in repr_str
    assert "cv=5" in repr_str


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_refits_base_models(mock_cross_val_predict, mock_models, sample_data):
    """Test that base models are refit on full data."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    ensemble.fit(X, y)
    
    # Each base model should be fit once (refit on full data)
    for model in ensemble.base_models:
        assert model.fit.called


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_with_real_ridge(mock_cross_val_predict, mock_models, sample_data):
    """Test stacking with real Ridge meta-learner."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    meta_learner = Ridge(alpha=1.0)
    
    ensemble = StackingEnsemble(mock_models, meta_learner, cv=5)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    assert predictions.shape == (100,)


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_different_cv_values(mock_cross_val_predict, mock_models, sample_data):
    """Test stacking with different cv values."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    for cv in [3, 5, 10]:
        ensemble = StackingEnsemble(mock_models, Ridge(), cv=cv)
        ensemble.fit(X, y)
        assert ensemble.cv == cv


@patch("sklearn.model_selection.cross_val_predict")
def test_stacking_save_and_load(
    mock_cross_val_predict, mock_models, sample_data, tmp_path
):
    """Test save and load stacking ensemble."""
    X, y = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    ensemble = StackingEnsemble(mock_models, Ridge(), cv=5)
    ensemble.fit(X, y)
    
    save_path = tmp_path / "stacking.joblib"
    ensemble.save(save_path)
    
    loaded = StackingEnsemble.load(save_path)
    assert loaded.cv == 5
    assert loaded._is_fitted is True