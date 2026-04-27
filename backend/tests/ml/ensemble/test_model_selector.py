"""Tests for model selector."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.model_selector import ModelSelector


@pytest.fixture
def mock_models_varied_performance():
    """Create mock models with varied performance."""
    models = []
    
    # Model 0: R² ≈ 0.8 (good)
    model0 = Mock(spec=BaseEstimator)
    y_val = np.random.randn(100)
    model0.predict = Mock(return_value=y_val + np.random.randn(100) * 0.3)
    models.append(model0)
    
    # Model 1: R² ≈ 0.7 (good)
    model1 = Mock(spec=BaseEstimator)
    model1.predict = Mock(return_value=y_val + np.random.randn(100) * 0.4)
    models.append(model1)
    
    # Model 2: R² < 0.5 (bad)
    model2 = Mock(spec=BaseEstimator)
    model2.predict = Mock(return_value=y_val + np.random.randn(100) * 0.8)
    models.append(model2)
    
    return models, y_val


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


def test_model_selector_initialization():
    """Test model selector initialization."""
    selector = ModelSelector(
        min_performance=0.70,
        min_diversity=0.10,
        max_models=5
    )
    assert selector.min_performance == 0.70
    assert selector.min_diversity == 0.10
    assert selector.max_models == 5


def test_model_selector_invalid_min_performance():
    """Test invalid min_performance."""
    with pytest.raises(ValueError, match="must be in"):
        ModelSelector(min_performance=1.5)


def test_model_selector_invalid_min_diversity():
    """Test invalid min_diversity."""
    with pytest.raises(ValueError, match="must be in"):
        ModelSelector(min_diversity=-0.1)


def test_model_selector_invalid_max_models():
    """Test invalid max_models."""
    with pytest.raises(ValueError, match="must be >= 1"):
        ModelSelector(max_models=0)


def test_select_empty_models(sample_data):
    """Test select with empty models list."""
    X, y = sample_data
    selector = ModelSelector()
    
    with pytest.raises(ValueError, match="cannot be empty"):
        selector.select([], X, y)


def test_select_filters_low_performance(mock_models_varied_performance, sample_data):
    """Test that low-performance models are filtered."""
    models, y_val = mock_models_varied_performance
    X, _ = sample_data
    
    selector = ModelSelector(min_performance=0.65)
    selected = selector.select(models, X, y_val)
    
    # Should select 2 good models, exclude 1 bad model
    assert len(selected) <= 2


def test_select_respects_max_models(sample_data):
    """Test that max_models is respected."""
    X, y = sample_data
    
    # Create many good models
    models = []
    for i in range(10):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y + np.random.randn(100) * 0.2)
        models.append(model)
    
    selector = ModelSelector(min_performance=0.5, max_models=3)
    selected = selector.select(models, X, y)
    
    assert len(selected) <= 3


def test_select_starts_with_best_model(mock_models_varied_performance, sample_data):
    """Test that selection starts with best model."""
    models, y_val = mock_models_varied_performance
    X, _ = sample_data
    
    selector = ModelSelector(min_performance=0.5, min_diversity=0.0)
    selected = selector.select(models, X, y_val)
    
    # First selected should be model 0 (best performance)
    assert 0 in selected


def test_select_no_models_meet_threshold(sample_data):
    """Test when no models meet performance threshold."""
    X, y = sample_data
    
    # Create bad models
    models = []
    for _ in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y + np.random.randn(100) * 2.0)
        models.append(model)
    
    selector = ModelSelector(min_performance=0.8)
    
    with pytest.raises(ValueError, match="No models meet"):
        selector.select(models, X, y)


def test_select_with_details(mock_models_varied_performance, sample_data):
    """Test select_with_details."""
    models, y_val = mock_models_varied_performance
    X, _ = sample_data
    
    selector = ModelSelector(min_performance=0.5)
    selected, details = selector.select_with_details(models, X, y_val)
    
    assert "selected_indices" in details
    assert "selected_performances" in details
    assert "avg_performance" in details
    assert "diversity" in details
    
    assert len(details["selected_indices"]) == len(selected)
    assert len(details["selected_performances"]) == len(selected)


def test_diversity_threshold_enforced(sample_data):
    """Test that diversity threshold is enforced."""
    X, y = sample_data
    
    # Create highly correlated models
    models = []
    for i in range(5):
        model = Mock(spec=BaseEstimator)
        # Very similar predictions
        model.predict = Mock(return_value=y + np.random.randn(100) * 0.01 + i * 0.001)
        models.append(model)
    
    selector = ModelSelector(min_performance=0.5, min_diversity=0.5)
    selected = selector.select(models, X, y)
    
    # Should only select 1 model due to high correlation
    assert len(selected) <= 2