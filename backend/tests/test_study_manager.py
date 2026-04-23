"""Tests for Optuna study manager."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bufferiq.ml.optimization.study_manager import OptunaStudyManager


@pytest.fixture
def manager(tmp_path):
    """Create study manager with temp storage."""
    storage = f"sqlite:///{tmp_path}/test.db"
    return OptunaStudyManager(storage)


def test_manager_initialization():
    """Test study manager initialization."""
    manager = OptunaStudyManager("sqlite:///test.db")
    assert manager.storage == "sqlite:///test.db"


@patch("optuna.create_study")
def test_create_study(mock_create_study, manager):
    """Test study creation."""
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    
    study = manager.create_study("test_study", direction="maximize")
    
    assert study == mock_study
    assert mock_create_study.called


@patch("optuna.create_study")
def test_create_duplicate_study_raises_error(mock_create_study, manager):
    """Test creating duplicate study raises error."""
    import optuna
    
    mock_create_study.side_effect = optuna.exceptions.DuplicatedStudyError()
    
    with pytest.raises(ValueError, match="already exists"):
        manager.create_study("duplicate_study")


@patch("optuna.load_study")
def test_load_study(mock_load_study, manager):
    """Test study loading."""
    mock_study = MagicMock()
    mock_study.trials = [MagicMock()]
    mock_load_study.return_value = mock_study
    
    study = manager.load_study("existing_study")
    
    assert study == mock_study
    assert mock_load_study.called


@patch("optuna.load_study")
def test_load_nonexistent_study_raises_error(mock_load_study, manager):
    """Test loading nonexistent study raises error."""
    mock_load_study.side_effect = KeyError("Study not found")
    
    with pytest.raises(ValueError, match="not found"):
        manager.load_study("nonexistent_study")


@patch("optuna.study.get_all_study_names")
def test_list_studies(mock_get_names, manager):
    """Test listing studies."""
    mock_get_names.return_value = ["study1", "study2", "study3"]
    
    studies = manager.list_studies()
    
    assert len(studies) == 3
    assert "study1" in studies


@patch("optuna.study.get_all_study_names")
def test_list_studies_handles_exception(mock_get_names, manager):
    """Test list_studies handles exceptions."""
    mock_get_names.side_effect = RuntimeError("Test error")
    
    studies = manager.list_studies()
    
    assert studies == []


@patch("optuna.delete_study")
def test_delete_study(mock_delete, manager):
    """Test study deletion."""
    manager.delete_study("study_to_delete")
    
    assert mock_delete.called


@patch("optuna.delete_study")
def test_delete_nonexistent_study(mock_delete, manager):
    """Test deleting nonexistent study."""
    mock_delete.side_effect = KeyError("Study not found")
    
    # Should not raise, just log warning
    manager.delete_study("nonexistent_study")


@patch("optuna.load_study")
def test_export_study(mock_load_study, manager, tmp_path):
    """Test study export."""
    # Setup mock study
    mock_study = MagicMock()
    mock_study.study_name = "test_study"
    mock_study.direction.name = "MAXIMIZE"
    mock_study.best_params = {"x": 1.0}
    mock_study.best_value = 0.95
    
    mock_trial = MagicMock()
    mock_trial.number = 0
    mock_trial.params = {"x": 1.0}
    mock_trial.value = 0.95
    mock_trial.state.name = "COMPLETE"
    mock_trial.datetime_start = None
    mock_trial.datetime_complete = None
    
    mock_study.trials = [mock_trial]
    mock_load_study.return_value = mock_study
    
    # Export
    save_path = tmp_path / "export.json"
    manager.export_study("test_study", save_path)
    
    assert save_path.exists()
    
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert data["study_name"] == "test_study"
    assert data["best_value"] == 0.95
    assert len(data["trials"]) == 1


@patch("optuna.load_study")
def test_get_study_summary(mock_load_study, manager):
    """Test getting study summary."""
    # Setup mock study
    mock_study = MagicMock()
    mock_study.study_name = "test_study"
    mock_study.direction.name = "MAXIMIZE"
    mock_study.best_params = {"x": 1.0}
    mock_study.best_value = 0.95
    
    import optuna
    
    mock_study.trials = [
        MagicMock(state=optuna.trial.TrialState.COMPLETE),
        MagicMock(state=optuna.trial.TrialState.COMPLETE),
        MagicMock(state=optuna.trial.TrialState.PRUNED),
        MagicMock(state=optuna.trial.TrialState.FAIL),
    ]
    
    mock_load_study.return_value = mock_study
    
    summary = manager.get_study_summary("test_study")
    
    assert summary["study_name"] == "test_study"
    assert summary["best_value"] == 0.95
    assert summary["n_trials"] == 4
    assert summary["n_complete"] == 2
    assert summary["n_pruned"] == 1
    assert summary["n_failed"] == 1