"""Tests for ensemble builder."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.ensemble_builder import EnsembleBuilder


@pytest.fixture
def mock_model_paths(tmp_path):
    """Create mock model paths."""
    paths = []
    for i in range(3):
        path = tmp_path / f"model_{i}.joblib"
        # Create dummy files
        path.touch()
        paths.append(path)
    return paths


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(50, 10)
    y_val = np.random.randn(50)
    return X_train, y_train, X_val, y_val


def test_ensemble_builder_initialization(mock_model_paths):
    """Test ensemble builder initialization."""
    builder = EnsembleBuilder(
        model_paths=mock_model_paths,
        ensemble_type="stacking"
    )
    assert len(builder.model_paths) == 3
    assert builder.ensemble_type == "stacking"


def test_ensemble_builder_empty_paths():
    """Test ensemble builder with empty paths."""
    with pytest.raises(ValueError, match="cannot be empty"):
        EnsembleBuilder(model_paths=[])


def test_ensemble_builder_invalid_type(mock_model_paths):
    """Test ensemble builder with invalid type."""
    with pytest.raises(ValueError, match="must be one of"):
        EnsembleBuilder(
            model_paths=mock_model_paths,
            ensemble_type="invalid"
        )


@patch("joblib.load")
def test_load_models(mock_joblib_load, mock_model_paths):
    """Test load_models."""
    # Mock loaded models
    mock_models = []
    for _ in range(3):
        model = Mock(spec=BaseEstimator)
        mock_models.append(model)
    
    mock_joblib_load.side_effect = mock_models
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    loaded = builder.load_models()
    
    assert len(loaded) == 3
    assert mock_joblib_load.call_count == 3


def test_load_models_file_not_found(tmp_path):
    """Test load_models with missing file."""
    paths = [tmp_path / "nonexistent.joblib"]
    builder = EnsembleBuilder(model_paths=paths)
    
    with pytest.raises(FileNotFoundError):
        builder.load_models()


@patch("joblib.load")
def test_analyze_diversity(mock_joblib_load, mock_model_paths, sample_data):
    """Test analyze_diversity."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(50))
        mock_models.append(model)
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    metrics = builder.analyze_diversity(mock_models, X_val, y_val)
    
    assert "correlation_diversity" in metrics
    assert "disagreement_diversity" in metrics
    assert "avg_q_statistic" in metrics


@patch("joblib.load")
def test_select_models(mock_joblib_load, mock_model_paths, sample_data):
    """Test select_models."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Mock models with different performances
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        # Model predictions with varying quality
        model.predict = Mock(return_value=y_val + np.random.randn(50) * (0.3 + i * 0.1))
        mock_models.append(model)
    
    builder = EnsembleBuilder(
        model_paths=mock_model_paths,
        min_performance=0.5
    )
    selected = builder.select_models(mock_models, X_val, y_val)
    
    assert len(selected) <= 3
    assert len(selected) >= 1


@patch("joblib.load")
@patch("sklearn.model_selection.cross_val_predict")
def test_build_voting(
    mock_cross_val_predict, mock_joblib_load, mock_model_paths, sample_data
):
    """Test build_voting."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(100))
        mock_models.append(model)
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    ensemble = builder.build_voting(mock_models, X_train, y_train)
    
    assert ensemble is not None
    assert ensemble._is_fitted is True


@patch("joblib.load")
@patch("sklearn.model_selection.cross_val_predict")
def test_build_stacking(
    mock_cross_val_predict, mock_joblib_load, mock_model_paths, sample_data
):
    """Test build_stacking."""
    X_train, y_train, X_val, y_val = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(100))
        model.fit = Mock(return_value=None)
        mock_models.append(model)
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    ensemble = builder.build_stacking(mock_models, X_train, y_train)
    
    assert ensemble is not None
    assert ensemble._is_fitted is True


@patch("joblib.load")
def test_build_blending(mock_joblib_load, mock_model_paths, sample_data):
    """Test build_blending."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(70))
        model.fit = Mock(return_value=None)
        mock_models.append(model)
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    ensemble = builder.build_blending(mock_models, X_train, y_train)
    
    assert ensemble is not None
    assert ensemble._is_fitted is True


@patch("joblib.load")
def test_build_weighted_average(mock_joblib_load, mock_model_paths, sample_data):
    """Test build_weighted_average."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=np.random.randn(100))
        mock_models.append(model)
    
    builder = EnsembleBuilder(model_paths=mock_model_paths)
    ensemble = builder.build_weighted_average(mock_models, X_train, y_train)
    
    assert ensemble is not None
    assert ensemble._is_fitted is True


@patch("joblib.load")
@patch("sklearn.model_selection.cross_val_predict")
def test_build_full_pipeline(
    mock_cross_val_predict, mock_joblib_load, mock_model_paths, sample_data
):
    """Test full build pipeline."""
    X_train, y_train, X_val, y_val = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y_val + np.random.randn(50) * 0.3)
        model.fit = Mock(return_value=None)
        mock_models.append(model)
    
    mock_joblib_load.side_effect = mock_models
    
    builder = EnsembleBuilder(
        model_paths=mock_model_paths,
        ensemble_type="voting",
        min_performance=0.5
    )
    
    ensemble = builder.build(X_train, y_train, X_val, y_val)
    
    assert ensemble is not None
    assert ensemble._is_fitted is True


@patch("joblib.load")
@patch("sklearn.model_selection.cross_val_predict")
def test_auto_select_ensemble_type(
    mock_cross_val_predict, mock_joblib_load, mock_model_paths, sample_data
):
    """Test auto ensemble type selection."""
    X_train, y_train, X_val, y_val = sample_data
    
    mock_cross_val_predict.return_value = np.random.randn(100)
    
    # Mock models
    mock_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y_val + np.random.randn(50) * 0.3)
        model.fit = Mock(return_value=None)
        mock_models.append(model)
    
    mock_joblib_load.side_effect = mock_models
    
    builder = EnsembleBuilder(
        model_paths=mock_model_paths,
        ensemble_type="auto",
        min_performance=0.5
    )
    
    ensemble = builder.build(X_train, y_train, X_val, y_val)
    
    assert ensemble is not None