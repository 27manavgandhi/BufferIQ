"""Tests for checkpoint manager."""

import tempfile
from pathlib import Path

import joblib
import pytest

from bufferiq.ml.training.checkpoint import Checkpoint


class TestCheckpoint:
    """Test checkpoint manager."""

    @pytest.fixture
    def temp_dir(self) -> str:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def checkpoint(self, temp_dir: str) -> Checkpoint:
        """Create checkpoint manager."""
        return Checkpoint(
            checkpoint_dir=temp_dir,
            monitor="val_loss",
            mode="min",
            patience=3,
        )

    def test_init(self, checkpoint: Checkpoint) -> None:
        """Test initialization."""
        assert checkpoint.monitor == "val_loss"
        assert checkpoint.mode == "min"
        assert checkpoint.patience == 3
        assert checkpoint.wait == 0

    def test_on_epoch_end_improvement(
        self, checkpoint: Checkpoint, temp_dir: str
    ) -> None:
        """Test on_epoch_end with improvement."""
        metrics = {"val_loss": 0.5}
        model = {"weights": [1, 2, 3]}

        should_stop = checkpoint.on_epoch_end(1, metrics, model)

        assert not should_stop
        assert checkpoint.best_value == 0.5
        assert checkpoint.wait == 0

        # Check checkpoint saved
        checkpoint_files = list(Path(temp_dir).glob("checkpoint_*.joblib"))
        assert len(checkpoint_files) == 1

    def test_on_epoch_end_no_improvement(
        self, checkpoint: Checkpoint
    ) -> None:
        """Test on_epoch_end without improvement."""
        model = {"weights": [1, 2, 3]}

        # First epoch (improvement)
        checkpoint.on_epoch_end(1, {"val_loss": 0.5}, model)

        # Second epoch (no improvement)
        should_stop = checkpoint.on_epoch_end(2, {"val_loss": 0.6}, model)

        assert not should_stop
        assert checkpoint.wait == 1

    def test_early_stopping_triggered(self, checkpoint: Checkpoint) -> None:
        """Test early stopping trigger."""
        model = {"weights": [1, 2, 3]}

        # First epoch - improvement
        checkpoint.on_epoch_end(1, {"val_loss": 0.5}, model)

        # No improvement for patience epochs
        for epoch in range(2, 2 + checkpoint.patience):
            should_stop = checkpoint.on_epoch_end(epoch, {"val_loss": 0.6}, model)

        # Should trigger early stopping
        assert should_stop
        assert checkpoint.stopped_epoch == 2 + checkpoint.patience - 1

    def test_mode_max(self, temp_dir: str) -> None:
        """Test checkpoint with max mode."""
        checkpoint = Checkpoint(
            checkpoint_dir=temp_dir, monitor="val_acc", mode="max", patience=3
        )

        model = {"weights": [1, 2, 3]}

        # First epoch
        checkpoint.on_epoch_end(1, {"val_acc": 0.8}, model)
        assert checkpoint.best_value == 0.8

        # Improvement (higher is better)
        checkpoint.on_epoch_end(2, {"val_acc": 0.9}, model)
        assert checkpoint.best_value == 0.9
        assert checkpoint.wait == 0

        # No improvement (lower value)
        checkpoint.on_epoch_end(3, {"val_acc": 0.85}, model)
        assert checkpoint.wait == 1

    def test_min_delta(self, temp_dir: str) -> None:
        """Test minimum delta for improvement."""
        checkpoint = Checkpoint(
            checkpoint_dir=temp_dir,
            monitor="val_loss",
            mode="min",
            min_delta=0.01,
            patience=3,
        )

        model = {"weights": [1, 2, 3]}

        checkpoint.on_epoch_end(1, {"val_loss": 0.5}, model)

        # Small improvement (less than min_delta)
        checkpoint.on_epoch_end(2, {"val_loss": 0.499}, model)
        assert checkpoint.wait == 1  # Not considered improvement

        # Large improvement (more than min_delta)
        checkpoint.on_epoch_end(3, {"val_loss": 0.48}, model)
        assert checkpoint.wait == 0  # Considered improvement

    def test_restore_best_model(self, checkpoint: Checkpoint) -> None:
        """Test restoring best model."""
        model1 = {"weights": [1, 2, 3]}
        model2 = {"weights": [4, 5, 6]}

        checkpoint.on_epoch_end(1, {"val_loss": 0.5}, model1)
        checkpoint.on_epoch_end(2, {"val_loss": 0.6}, model2)

        best_model = checkpoint.restore_best_model()

        assert best_model == model1

    def test_restore_best_model_no_checkpoint(
        self, checkpoint: Checkpoint
    ) -> None:
        """Test restore with no checkpoint raises error."""
        with pytest.raises(ValueError, match="No checkpoint available"):
            checkpoint.restore_best_model()

    def test_get_best_metrics(self, checkpoint: Checkpoint) -> None:
        """Test getting best metrics."""
        model = {"weights": [1, 2, 3]}

        checkpoint.on_epoch_end(1, {"val_loss": 0.5, "val_acc": 0.8}, model)
        checkpoint.on_epoch_end(2, {"val_loss": 0.6, "val_acc": 0.85}, model)

        best_metrics = checkpoint.get_best_metrics()

        assert best_metrics["val_loss"] == 0.5
        assert best_metrics["val_acc"] == 0.8

    def test_missing_metric_raises_error(self, checkpoint: Checkpoint) -> None:
        """Test missing monitored metric raises error."""
        model = {"weights": [1, 2, 3]}

        with pytest.raises(ValueError, match="Monitored metric.*not found"):
            checkpoint.on_epoch_end(1, {"other_metric": 0.5}, model)