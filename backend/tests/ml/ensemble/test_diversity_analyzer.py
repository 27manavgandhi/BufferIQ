"""Tests for diversity analyzer."""

from pathlib import Path

import numpy as np
import pytest

from bufferiq.ml.ensemble.diversity_analyzer import DiversityAnalyzer


@pytest.fixture
def identical_predictions():
    """Create identical predictions."""
    pred = np.array([1, 2, 3, 4, 5])
    return np.column_stack([pred, pred, pred])


@pytest.fixture
def diverse_predictions():
    """Create diverse predictions."""
    pred1 = np.array([1, 2, 3, 4, 5])
    pred2 = np.array([5, 4, 3, 2, 1])
    pred3 = np.array([2, 3, 4, 5, 1])
    return np.column_stack([pred1, pred2, pred3])


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    np.random.seed(42)
    return np.random.randn(100, 3)


def test_correlation_diversity_identical(identical_predictions):
    """Test correlation diversity with identical predictions."""
    diversity = DiversityAnalyzer.correlation_diversity(identical_predictions)
    # Identical predictions should have correlation ≈ 1, diversity ≈ 0
    assert diversity < 0.1


def test_correlation_diversity_diverse(diverse_predictions):
    """Test correlation diversity with diverse predictions."""
    diversity = DiversityAnalyzer.correlation_diversity(diverse_predictions)
    # Diverse predictions should have higher diversity
    assert diversity > 0.5


def test_correlation_diversity_wrong_shape():
    """Test correlation diversity with wrong shape."""
    predictions = np.array([1, 2, 3])  # 1D instead of 2D
    
    with pytest.raises(ValueError, match="must be 2D"):
        DiversityAnalyzer.correlation_diversity(predictions)


def test_correlation_diversity_single_model():
    """Test correlation diversity with single model."""
    predictions = np.random.randn(100, 1)
    
    with pytest.raises(ValueError, match="at least 2 models"):
        DiversityAnalyzer.correlation_diversity(predictions)


def test_disagreement_diversity_identical(identical_predictions):
    """Test disagreement diversity with identical predictions."""
    diversity = DiversityAnalyzer.disagreement_diversity(identical_predictions)
    # Identical predictions should have disagreement ≈ 0
    assert diversity < 0.1


def test_disagreement_diversity_diverse(diverse_predictions):
    """Test disagreement diversity with diverse predictions."""
    diversity = DiversityAnalyzer.disagreement_diversity(diverse_predictions)
    # Diverse predictions should have higher disagreement
    assert diversity > 0.5


def test_disagreement_diversity_custom_threshold(diverse_predictions):
    """Test disagreement diversity with custom threshold."""
    diversity_low = DiversityAnalyzer.disagreement_diversity(
        diverse_predictions, threshold=0.01
    )
    diversity_high = DiversityAnalyzer.disagreement_diversity(
        diverse_predictions, threshold=10.0
    )
    
    # Higher threshold should result in lower disagreement
    assert diversity_low > diversity_high


def test_q_statistic_shape(sample_predictions):
    """Test Q-statistic matrix shape."""
    y_true = np.random.randn(100)
    q_matrix = DiversityAnalyzer.q_statistic(sample_predictions, y_true)
    
    assert q_matrix.shape == (3, 3)
    # Diagonal should be 0
    assert np.allclose(np.diag(q_matrix), 0)


def test_q_statistic_symmetric(sample_predictions):
    """Test Q-statistic matrix is symmetric."""
    y_true = np.random.randn(100)
    q_matrix = DiversityAnalyzer.q_statistic(sample_predictions, y_true)
    
    assert np.allclose(q_matrix, q_matrix.T)


def test_q_statistic_length_mismatch():
    """Test Q-statistic with length mismatch."""
    predictions = np.random.randn(100, 3)
    y_true = np.random.randn(50)  # Wrong length
    
    with pytest.raises(ValueError, match="same length"):
        DiversityAnalyzer.q_statistic(predictions, y_true)


def test_visualize_correlation_matrix(sample_predictions, tmp_path):
    """Test correlation matrix visualization."""
    model_names = ["Model1", "Model2", "Model3"]
    save_path = tmp_path / "correlation.png"
    
    DiversityAnalyzer.visualize_correlation_matrix(
        sample_predictions, model_names, save_path
    )
    
    assert save_path.exists()


def test_visualize_disagreement_matrix(sample_predictions, tmp_path):
    """Test disagreement matrix visualization."""
    model_names = ["Model1", "Model2", "Model3"]
    save_path = tmp_path / "disagreement.png"
    
    DiversityAnalyzer.visualize_disagreement_matrix(
        sample_predictions, model_names, save_path
    )
    
    assert save_path.exists()


def test_analyze_all(sample_predictions, tmp_path):
    """Test comprehensive diversity analysis."""
    y_true = np.random.randn(100)
    model_names = ["Model1", "Model2", "Model3"]
    output_dir = tmp_path / "diversity"
    
    metrics = DiversityAnalyzer.analyze_all(
        sample_predictions, y_true, model_names, output_dir
    )
    
    assert "correlation_diversity" in metrics
    assert "disagreement_diversity" in metrics
    assert "avg_q_statistic" in metrics
    
    # Check visualizations created
    assert (output_dir / "correlation_matrix.png").exists()
    assert (output_dir / "disagreement_matrix.png").exists()


def test_correlation_diversity_bounds(sample_predictions):
    """Test correlation diversity is in valid range."""
    diversity = DiversityAnalyzer.correlation_diversity(sample_predictions)
    assert 0.0 <= diversity <= 2.0  # Can be > 1 if correlations are negative


def test_disagreement_diversity_bounds(sample_predictions):
    """Test disagreement diversity is in valid range."""
    diversity = DiversityAnalyzer.disagreement_diversity(sample_predictions)
    assert 0.0 <= diversity <= 1.0