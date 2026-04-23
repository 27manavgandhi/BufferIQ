"""Tests for hyperparameter importance analyzer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bufferiq.ml.optimization.param_importance import (
    HyperparameterImportanceAnalyzer,
)


@pytest.fixture
def mock_study():
    """Create mock Optuna study."""
    study = MagicMock()
    study.trials = [MagicMock() for _ in range(10)]
    return study


def test_analyzer_initialization(mock_study):
    """Test analyzer initialization."""
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    assert analyzer.study == mock_study


def test_analyzer_empty_study_raises_error():
    """Test empty study raises error."""
    study = MagicMock()
    study.trials = []
    
    with pytest.raises(ValueError, match="no trials"):
        HyperparameterImportanceAnalyzer(study)


@patch("optuna.importance.get_param_importances")
def test_calculate_importance(mock_get_importance, mock_study):
    """Test importance calculation."""
    mock_get_importance.return_value = {
        "learning_rate": 0.45,
        "max_depth": 0.32,
        "n_estimators": 0.15,
    }
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    importance = analyzer.calculate_importance()
    
    assert "learning_rate" in importance
    assert importance["learning_rate"] == 0.45
    assert mock_get_importance.called


@patch("optuna.importance.get_param_importances")
def test_calculate_importance_with_target(mock_get_importance, mock_study):
    """Test importance calculation with target objective."""
    mock_get_importance.return_value = {"param": 0.5}
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    importance = analyzer.calculate_importance(target=0)
    
    assert mock_get_importance.called
    call_kwargs = mock_get_importance.call_args[1]
    assert call_kwargs["target"] == 0


def test_unknown_method_raises_error(mock_study):
    """Test unknown method raises error."""
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.calculate_importance(method="invalid")


@patch("matplotlib.pyplot.savefig")
def test_visualize_importance(mock_savefig, mock_study, tmp_path):
    """Test importance visualization."""
    importance = {
        "learning_rate": 0.45,
        "max_depth": 0.32,
        "n_estimators": 0.15,
        "subsample": 0.08,
    }
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    save_path = tmp_path / "importance.png"
    analyzer.visualize_importance(importance, save_path)
    
    assert mock_savefig.called


@patch("matplotlib.pyplot.savefig")
def test_visualize_importance_top_n(mock_savefig, mock_study, tmp_path):
    """Test visualize importance with top_n."""
    importance = {f"param_{i}": 0.1 * i for i in range(20)}
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    save_path = tmp_path / "importance.png"
    analyzer.visualize_importance(importance, save_path, top_n=5)
    
    assert mock_savefig.called


@patch("matplotlib.pyplot.savefig")
def test_visualize_empty_importance(mock_savefig, mock_study, tmp_path):
    """Test visualize with empty importance."""
    importance = {}
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    save_path = tmp_path / "importance.png"
    analyzer.visualize_importance(importance, save_path)
    
    # Should not crash, but may not call savefig
    assert not mock_savefig.called


def test_export_rankings(mock_study, tmp_path):
    """Test export importance rankings."""
    importance = {
        "learning_rate": 0.45,
        "max_depth": 0.32,
        "n_estimators": 0.15,
    }
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    save_path = tmp_path / "rankings.json"
    analyzer.export_rankings(importance, save_path)
    
    assert save_path.exists()
    
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert "rankings" in data
    assert "total_parameters" in data
    assert len(data["rankings"]) == 3
    assert data["rankings"][0]["parameter"] == "learning_rate"
    assert data["rankings"][0]["rank"] == 1


@patch("optuna.importance.get_param_importances")
def test_calculate_importance_handles_exception(mock_get_importance, mock_study):
    """Test importance calculation handles exceptions."""
    mock_get_importance.side_effect = RuntimeError("Test error")
    
    analyzer = HyperparameterImportanceAnalyzer(mock_study)
    
    with pytest.raises(RuntimeError):
        analyzer.calculate_importance()