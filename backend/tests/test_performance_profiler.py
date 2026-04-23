"""Tests for performance profiler."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from bufferiq.ml.optimization.performance_profiler import PerformanceProfiler


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randn(200)
    return X, y


def test_profiler_initialization():
    """Test profiler initialization."""
    profiler = PerformanceProfiler()
    assert profiler.profiles == []


def test_profile_training(sample_data):
    """Test training profiling."""
    X, y = sample_data
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    
    profiler = PerformanceProfiler()
    profile = profiler.profile_training(model, X, y)
    
    assert "training_time" in profile
    assert "inference_time" in profile
    assert "predictions_per_second" in profile
    assert "model_size_mb" in profile
    
    assert profile["training_time"] > 0
    assert profile["predictions_per_second"] > 0
    assert len(profiler.profiles) == 1


def test_profile_multiple_models(sample_data):
    """Test profiling multiple models."""
    X, y = sample_data
    
    profiler = PerformanceProfiler()
    
    for n_estimators in [5, 10, 15]:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        profiler.profile_training(model, X, y)
    
    assert len(profiler.profiles) == 3


def test_estimate_model_size_tree_model(sample_data):
    """Test model size estimation for tree-based model."""
    X, y = sample_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    profiler = PerformanceProfiler()
    profile = profiler.profile_training(model, X, y)
    
    # Should be roughly n_estimators * 0.01
    assert profile["model_size_mb"] > 0
    assert profile["model_size_mb"] < 1.0


@patch("matplotlib.pyplot.savefig")
def test_visualize_performance(mock_savefig, sample_data, tmp_path):
    """Test performance visualization."""
    X, y = sample_data
    
    profiler = PerformanceProfiler()
    
    # Profile multiple models
    accuracy_scores = []
    for n_estimators in [5, 10, 15]:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        profiler.profile_training(model, X, y)
        accuracy_scores.append(0.7 + n_estimators * 0.01)
    
    save_path = tmp_path / "performance.png"
    profiler.visualize_performance(accuracy_scores, save_path)
    
    assert mock_savefig.called


@patch("matplotlib.pyplot.savefig")
def test_visualize_empty_profiles(mock_savefig, tmp_path):
    """Test visualize with no profiles."""
    profiler = PerformanceProfiler()
    save_path = tmp_path / "performance.png"
    
    profiler.visualize_performance([], save_path)
    
    # Should not call savefig
    assert not mock_savefig.called


def test_visualize_mismatched_lengths(sample_data, tmp_path):
    """Test visualize with mismatched array lengths."""
    X, y = sample_data
    
    profiler = PerformanceProfiler()
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    profiler.profile_training(model, X, y)
    
    save_path = tmp_path / "performance.png"
    
    with pytest.raises(ValueError, match="Length mismatch"):
        profiler.visualize_performance([0.7, 0.8], save_path)


def test_export_profiles(sample_data, tmp_path):
    """Test exporting performance profiles."""
    X, y = sample_data
    
    profiler = PerformanceProfiler()
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    profiler.profile_training(model, X, y)
    
    save_path = tmp_path / "profiles.json"
    profiler.export_profiles(save_path)
    
    assert save_path.exists()
    
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert "n_profiles" in data
    assert "profiles" in data
    assert data["n_profiles"] == 1


def test_export_empty_profiles(tmp_path):
    """Test exporting empty profiles."""
    profiler = PerformanceProfiler()
    save_path = tmp_path / "profiles.json"
    profiler.export_profiles(save_path)
    
    assert save_path.exists()
    
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert data["n_profiles"] == 0
    assert data["profiles"] == []