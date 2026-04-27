"""Tests for blending ensemble."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from bufferiq.ml.ensemble.blending import BlendingEnsemble


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


def test_blending_ensemble_initialization(mock_models):
    """Test blending ensemble initialization."""
    meta_learner = Ridge()
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    
    assert len(ensemble.base_models) == 3
    assert ensemble.blend_split == 0.3


def test_blending_ensemble_empty_models():
    """Test blending with empty models list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        BlendingEnsemble([], Ridge())


def test_blending_ensemble_invalid_blend_split(mock_models):
    """Test blending with invalid blend_split."""
    with pytest.raises(ValueError, match="must be in"):
        BlendingEnsemble(mock_models, Ridge(), blend_split=1.5)
    
    with pytest.raises(ValueError, match="must be in"):
        BlendingEnsemble(mock_models, Ridge(), blend_split=0.0)


def test_blending_ensemble_fit(mock_models, sample_data):
    """Test blending ensemble fit."""
    X, y = sample_data
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    result = ensemble.fit(X, y)
    
    assert result is ensemble
    assert ensemble._is_fitted is True
    assert meta_learner.fit.called


def test_blending_ensemble_predict(mock_models, sample_data):
    """Test blending ensemble predict."""
    X, y = sample_data
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    meta_learner.predict = Mock(return_value=np.random.randn(100))
    
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    
    assert predictions.shape == (100,)
    assert meta_learner.predict.called


def test_predict_before_fit_raises_error(mock_models, sample_data):
    """Test predict before fit raises error."""
    X, y = sample_data
    ensemble = BlendingEnsemble(mock_models, Ridge())
    
    with pytest.raises(ValueError, match="must be fitted"):
        ensemble.predict(X)


def test_blending_ensemble_repr(mock_models):
    """Test string representation."""
    meta_learner = Ridge()
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    repr_str = repr(ensemble)
    
    assert "BlendingEnsemble" in repr_str
    assert "n_base_models=3" in repr_str
    assert "Ridge" in repr_str
    assert "blend_split=0.3" in repr_str


def test_blending_refits_base_models(mock_models, sample_data):
    """Test that base models are refit on full data."""
    X, y = sample_data
    
    meta_learner = Mock(spec=BaseEstimator)
    meta_learner.fit = Mock(return_value=None)
    
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    ensemble.fit(X, y)
    
    # Each base model should be fit twice:
    # Once on train set, once on full data
    for model in ensemble.base_models:
        assert model.fit.call_count == 2


def test_blending_with_real_ridge(mock_models, sample_data):
    """Test blending with real Ridge meta-learner."""
    X, y = sample_data
    
    meta_learner = Ridge(alpha=0.5)
    
    ensemble = BlendingEnsemble(mock_models, meta_learner, blend_split=0.3)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    assert predictions.shape == (100,)


def test_blending_different_splits(mock_models, sample_data):
    """Test blending with different blend splits."""
    X, y = sample_data
    
    for split in [0.2, 0.3, 0.4]:
        ensemble = BlendingEnsemble(mock_models, Ridge(), blend_split=split)
        ensemble.fit(X, y)
        assert ensemble.blend_split == split


def test_blending_save_and_load(mock_models, sample_data, tmp_path):
    """Test save and load blending ensemble."""
    X, y = sample_data
    
    ensemble = BlendingEnsemble(mock_models, Ridge(), blend_split=0.3)
    ensemble.fit(X, y)
    
    save_path = tmp_path / "blending.joblib"
    ensemble.save(save_path)
    
    loaded = BlendingEnsemble.load(save_path)
    assert loaded.blend_split == 0.3
    assert loaded._is_fitted is True


def test_blending_with_small_dataset():
    """Test blending with small dataset."""
    X = np.random.randn(20, 5)
    y = np.random.randn(20)
    
    models = []
    for _ in range(2):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(len(X)))
        model.fit = Mock(return_value=None)
        models.append(model)
    
    ensemble = BlendingEnsemble(models, Ridge(), blend_split=0.3)
    ensemble.fit(X, y)
    
    predictions = ensemble.predict(X)
    assert predictions.shape == (20,)