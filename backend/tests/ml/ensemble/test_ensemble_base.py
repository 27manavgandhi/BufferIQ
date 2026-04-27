"""Tests for base ensemble class."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from bufferiq.ml.ensemble.base import BaseEnsemble


class ConcreteEnsemble(BaseEnsemble):
    """Concrete implementation for testing."""
    
    def fit(self, X, y):
        self.validate_inputs(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        self.check_is_fitted()
        self.validate_inputs(X)
        return np.zeros(len(X))


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_base_ensemble_initialization():
    """Test base ensemble initialization."""
    ensemble = ConcreteEnsemble()
    assert ensemble._is_fitted is False


def test_validate_inputs_valid(sample_data):
    """Test validate_inputs with valid data."""
    X, y = sample_data
    ensemble = ConcreteEnsemble()
    ensemble.validate_inputs(X, y)  # Should not raise


def test_validate_inputs_empty_x():
    """Test validate_inputs with empty X."""
    ensemble = ConcreteEnsemble()
    X = np.array([])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Empty input X"):
        ensemble.validate_inputs(X, y)


def test_validate_inputs_wrong_shape():
    """Test validate_inputs with wrong shape."""
    ensemble = ConcreteEnsemble()
    X = np.array([1, 2, 3])  # 1D instead of 2D
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="must be 2D"):
        ensemble.validate_inputs(X, y)


def test_validate_inputs_length_mismatch():
    """Test validate_inputs with length mismatch."""
    ensemble = ConcreteEnsemble()
    X = np.random.randn(10, 5)
    y = np.random.randn(20)
    
    with pytest.raises(ValueError, match="same length"):
        ensemble.validate_inputs(X, y)


def test_check_is_fitted_before_fit():
    """Test check_is_fitted before fitting."""
    ensemble = ConcreteEnsemble()
    
    with pytest.raises(ValueError, match="must be fitted"):
        ensemble.check_is_fitted()


def test_check_is_fitted_after_fit(sample_data):
    """Test check_is_fitted after fitting."""
    X, y = sample_data
    ensemble = ConcreteEnsemble()
    ensemble.fit(X, y)
    ensemble.check_is_fitted()  # Should not raise


def test_save_and_load(sample_data, tmp_path):
    """Test save and load functionality."""
    X, y = sample_data
    
    # Fit ensemble
    ensemble = ConcreteEnsemble()
    ensemble.fit(X, y)
    
    # Save
    save_path = tmp_path / "ensemble.joblib"
    ensemble.save(save_path)
    
    assert save_path.exists()
    
    # Load
    loaded = BaseEnsemble.load(save_path)
    assert loaded._is_fitted is True


def test_load_nonexistent_file(tmp_path):
    """Test loading nonexistent file."""
    path = tmp_path / "nonexistent.joblib"
    
    with pytest.raises(FileNotFoundError):
        BaseEnsemble.load(path)


def test_get_params():
    """Test get_params."""
    ensemble = ConcreteEnsemble()
    params = ensemble.get_params()
    assert isinstance(params, dict)


def test_set_params():
    """Test set_params."""
    ensemble = ConcreteEnsemble()
    ensemble.set_params(custom_param=123)
    assert ensemble.custom_param == 123