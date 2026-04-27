"""Tests for ensemble performance comparator."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from bufferiq.ml.ensemble.performance_comparator import (
    EnsemblePerformanceComparator,
)


@pytest.fixture
def mock_ensemble_and_models():
    """Create mock ensemble and base models."""
    np.random.seed(42)
    y_test = np.random.randn(100)
    
    # Ensemble with better performance
    ensemble = Mock(spec=BaseEstimator)
    ensemble.predict = Mock(return_value=y_test + np.random.randn(100) * 0.2)
    
    # Base models with varying performance
    base_models = []
    for i in range(3):
        model = Mock(spec=BaseEstimator)
        model.predict = Mock(return_value=y_test + np.random.randn(100) * (0.3 + i * 0.1))
        base_models.append(model)
    
    return ensemble, base_models, y_test


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X_test = np.random.randn(100, 10)
    y_test = np.random.randn(100)
    return X_test, y_test


def test_compare_structure(mock_ensemble_and_models, sample_data):
    """Test compare returns correct structure."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    assert "ensemble_metrics" in results
    assert "base_metrics" in results
    assert "improvement_pct" in results
    assert "statistical_tests" in results


def test_compare_ensemble_metrics(mock_ensemble_and_models, sample_data):
    """Test ensemble metrics in comparison."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    ensemble_metrics = results["ensemble_metrics"]
    
    assert "r2" in ensemble_metrics
    assert "mae" in ensemble_metrics
    assert "rmse" in ensemble_metrics


def test_compare_base_metrics(mock_ensemble_and_models, sample_data):
    """Test base model metrics in comparison."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    base_metrics = results["base_metrics"]
    
    assert len(base_metrics) == 3
    for metrics in base_metrics:
        assert "name" in metrics
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics


def test_compare_statistical_tests(mock_ensemble_and_models, sample_data):
    """Test statistical tests in comparison."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    tests = results["statistical_tests"]
    
    assert len(tests) == 3
    for test in tests:
        assert "model_index" in test
        assert "paired_t_test" in test
        assert "wilcoxon_test" in test


def test_visualize_comparison(mock_ensemble_and_models, sample_data, tmp_path):
    """Test visualization generation."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    save_path = tmp_path / "comparison.png"
    comparator.visualize_comparison(results, save_path)
    
    assert save_path.exists()


def test_export_report(mock_ensemble_and_models, sample_data, tmp_path):
    """Test report export."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    save_path = tmp_path / "report.json"
    comparator.export_report(results, save_path)
    
    assert save_path.exists()


def test_improvement_calculation(mock_ensemble_and_models, sample_data):
    """Test improvement percentage calculation."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    improvement = results["improvement_pct"]
    
    # Ensemble should show some improvement
    assert isinstance(improvement, (int, float))


def test_statistical_significance(mock_ensemble_and_models, sample_data):
    """Test statistical significance detection."""
    ensemble, base_models, y_test = mock_ensemble_and_models
    X_test, _ = sample_data
    
    model_names = ["Model1", "Model2", "Model3"]
    
    comparator = EnsemblePerformanceComparator()
    results = comparator.compare(ensemble, base_models, X_test, y_test, model_names)
    
    for test in results["statistical_tests"]:
        t_test = test["paired_t_test"]
        wilcoxon = test["wilcoxon_test"]
        
        assert "statistic" in t_test
        assert "p_value" in t_test
        assert "significant" in t_test
        
        assert "statistic" in wilcoxon
        assert "p_value" in wilcoxon
        assert "significant" in wilcoxon