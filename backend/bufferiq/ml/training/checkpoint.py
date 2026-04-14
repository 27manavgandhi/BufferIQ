"""Model checkpointing and early stopping."""

from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class Checkpoint:
    """Model checkpointing and early stopping."""

    def __init__(
        self,
        checkpoint_dir: str = "outputs/models/checkpoints",
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 10,
        min_delta: float = 0.001,
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor (e.g., 'val_loss', 'val_r2')
            mode: 'min' to minimize metric, 'max' to maximize
            patience: Epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
            save_best_only: Only save best checkpoint
            verbose: Print checkpoint messages

        Example:
            >>> checkpoint = Checkpoint(monitor='val_r2', mode='max', patience=10)
            >>> for epoch in range(100):
            ...     metrics = train_epoch()
            ...     if checkpoint.on_epoch_end(epoch, metrics, model):
            ...         break  # Early stopping triggered
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_only = save_best_only
        self.verbose = verbose

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best_model_path: Optional[str] = None
        self.best_metrics: dict[str, float] = {}

        # Set comparison function
        if mode == "min":
            self.monitor_op = np.less
            self.best_value = np.inf
        else:
            self.monitor_op = np.greater
            self.best_value = -np.inf

    def should_stop(self) -> bool:
        """
        Check if training should stop (early stopping).

        Returns:
            True if should stop training
        """
        return self.wait >= self.patience

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: Any) -> bool:
        """
        Called at end of each epoch.

        Args:
            epoch: Current epoch number
            metrics: Dict of metric values
            model: Current model to checkpoint

        Returns:
            True if should stop training

        Raises:
            ValueError: If monitored metric not in metrics dict
        """
        if self.monitor not in metrics:
            raise ValueError(
                f"Monitored metric '{self.monitor}' not found in metrics. "
                f"Available: {list(metrics.keys())}"
            )

        current_value = metrics[self.monitor]

        # Check if this is an improvement
        if self.best_value is None:
            is_improvement = True
        else:
            if self.mode == "min":
                is_improvement = current_value < (self.best_value - self.min_delta)
            else:
                is_improvement = current_value > (self.best_value + self.min_delta)

        if is_improvement:
            # Update best value
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            self.best_metrics = metrics.copy()

            # Save checkpoint
            if not self.save_best_only or is_improvement:
                checkpoint_path = (
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.joblib"
                )
                joblib.dump(model, checkpoint_path)
                self.best_model_path = str(checkpoint_path)

                if self.verbose:
                    logger.info(
                        f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}, "
                        f"saving model to {checkpoint_path}"
                    )

        else:
            # No improvement
            self.wait += 1

            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor}={current_value:.4f}, "
                    f"no improvement for {self.wait} epochs "
                    f"(best: {self.best_value:.4f} at epoch {self.best_epoch})"
                )

            # Check for early stopping
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch}, "
                        f"best {self.monitor}: {self.best_value:.4f}"
                    )

                return True

        return False

    def restore_best_model(self) -> Any:
        """
        Load best checkpoint.

        Returns:
            Best model

        Raises:
            ValueError: If no checkpoint saved
        """
        if self.best_model_path is None:
            raise ValueError("No checkpoint available to restore")

        model = joblib.load(self.best_model_path)

        if self.verbose:
            logger.info(
                f"Restored best model from epoch {self.best_epoch} "
                f"with {self.monitor}={self.best_value:.4f}"
            )

        return model

    def get_best_metrics(self) -> dict[str, float]:
        """
        Get metrics from best checkpoint.

        Returns:
            Dict with best metrics
        """
        return self.best_metrics.copy()
