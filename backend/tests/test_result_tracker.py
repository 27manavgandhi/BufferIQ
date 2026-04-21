"""Tests for optimization result tracker."""

import json
from pathlib import Path

import pytest
import yaml

from bufferiq.ml.optimization.result_tracker import OptimizationResultTracker


@pytest.fixture
def tracker(tmp_path):
    """Create result tracker with temporary directory."""
    return OptimizationResultTracker(tmp_path)


def test_tracker_initialization(tmp_path):
    """Test tracker initialization."""
    tracker = OptimizationResultTracker(tmp_path)
    assert tracker.output_dir == tmp_path
    assert tracker.trials == []
    assert tracker.best_trial is None


def test_tracker_set_baseline(tracker):
    """Test setting baseline score."""
    tracker.set_baseline(0.70)
    assert tracker.baseline_score == 0.70


def test_tracker_log_trial(tracker):
    """Test logging a trial."""
    tracker.log_trial(
        trial_id=1,
        params={"learning_rate": 0.1},
        score=0.75,
        duration=10.5,
    )
    
    assert len(tracker.trials) == 1
    assert tracker.trials[0]["trial_id"] == 1
    assert tracker.trials[0]["score"] == 0.75


def test_tracker_get_best_trial_initially_none(tracker):
    """Test get_best_trial returns None initially."""
    assert tracker.get_best_trial() is None


def test_tracker_get_best_trial_after_logging(tracker):
    """Test get_best_trial returns best trial."""
    tracker.log_trial(1, {"lr": 0.1}, 0.70, 10.0)
    tracker.log_trial(2, {"lr": 0.2}, 0.75, 12.0)
    tracker.log_trial(3, {"lr": 0.05}, 0.72, 11.0)
    
    best = tracker.get_best_trial()
    assert best is not None
    assert best["trial_id"] == 2
    assert best["score"] == 0.75


def test_tracker_update_best_trial(tracker):
    """Test best trial updates when better score found."""
    tracker.log_trial(1, {"lr": 0.1}, 0.70, 10.0)
    assert tracker.get_best_trial()["trial_id"] == 1
    
    tracker.log_trial(2, {"lr": 0.2}, 0.80, 12.0)
    assert tracker.get_best_trial()["trial_id"] == 2
    
    tracker.log_trial(3, {"lr": 0.15}, 0.75, 11.0)
    assert tracker.get_best_trial()["trial_id"] == 2  # Still trial 2


def test_tracker_get_improvement_without_baseline(tracker):
    """Test get_improvement returns None without baseline."""
    tracker.log_trial(1, {"lr": 0.1}, 0.75, 10.0)
    assert tracker.get_improvement() is None


def test_tracker_get_improvement_with_baseline(tracker):
    """Test get_improvement calculates correctly."""
    tracker.set_baseline(0.70)
    tracker.log_trial(1, {"lr": 0.1}, 0.77, 10.0)
    
    improvement = tracker.get_improvement()
    assert improvement is not None
    assert abs(improvement - 10.0) < 0.01  # (0.77 - 0.70) / 0.70 * 100


def test_tracker_save_trials(tracker, tmp_path):
    """Test saving trials to JSON."""
    tracker.log_trial(1, {"lr": 0.1}, 0.75, 10.0)
    tracker.log_trial(2, {"lr": 0.2}, 0.78, 12.0)
    
    filepath = tracker.save_trials()
    
    assert filepath.exists()
    with open(filepath) as f:
        data = json.load(f)
    assert len(data) == 2


def test_tracker_export_best_params_without_trials(tracker):
    """Test export_best_params raises error without trials."""
    with pytest.raises(ValueError, match="No trials logged"):
        tracker.export_best_params()


def test_tracker_export_best_params(tracker, tmp_path):
    """Test exporting best parameters to YAML."""
    tracker.log_trial(1, {"lr": 0.1}, 0.75, 10.0)
    tracker.log_trial(2, {"lr": 0.2}, 0.78, 12.0)
    
    filepath = tracker.export_best_params()
    
    assert filepath.exists()
    with open(filepath) as f:
        data = yaml.safe_load(f)
    assert data["best_params"] == {"lr": 0.2}
    assert data["best_score"] == 0.78


def test_tracker_export_best_params_with_baseline(tracker):
    """Test export includes baseline and improvement."""
    tracker.set_baseline(0.70)
    tracker.log_trial(1, {"lr": 0.1}, 0.77, 10.0)
    
    filepath = tracker.export_best_params()
    
    with open(filepath) as f:
        data = yaml.safe_load(f)
    assert "baseline_score" in data
    assert "improvement_pct" in data


def test_tracker_generate_report_empty(tracker):
    """Test generate_report with no trials."""
    report = tracker.generate_report()
    assert report["status"] == "No trials logged"


def test_tracker_generate_report(tracker):
    """Test generate_report with trials."""
    tracker.log_trial(1, {"lr": 0.1}, 0.70, 10.0)
    tracker.log_trial(2, {"lr": 0.2}, 0.75, 12.0)
    tracker.log_trial(3, {"lr": 0.15}, 0.72, 11.0)
    
    report = tracker.generate_report()
    
    assert report["total_trials"] == 3
    assert report["best_score"] == 0.75
    assert report["best_trial_id"] == 2
    assert "mean_score" in report
    assert "std_score" in report


def test_tracker_save_report(tracker, tmp_path):
    """Test saving report to JSON."""
    tracker.log_trial(1, {"lr": 0.1}, 0.75, 10.0)
    
    filepath = tracker.save_report()
    
    assert filepath.exists()
    with open(filepath) as f:
        data = json.load(f)
    assert "total_trials" in data