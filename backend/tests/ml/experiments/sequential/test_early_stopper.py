"""Tests for early stopper."""

import numpy as np
from bufferiq.ml.experiments.sequential.early_stopper import EarlyStopper


class TestEarlyStopper:
    """Test EarlyStopper."""

    def setup_method(self):
        """Setup test."""
        self.stopper = EarlyStopper(min_samples_per_variant=100)

    def test_insufficient_samples(self):
        """Test insufficient samples check."""
        control = np.random.normal(100, 15, 50)
        treatment = np.random.normal(110, 15, 50)

        result = self.stopper.check_stopping_criteria(control, treatment)

        assert result["should_stop"] is False
        assert "Insufficient" in result["reason"]

    def test_clear_winner(self):
        """Test clear winner detection."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 500)
        treatment = np.random.normal(130, 15, 500)  # Large difference

        result = self.stopper.check_stopping_criteria(control, treatment)

        # Should detect clear winner
        assert result["should_stop"] is True
        assert result["reason"] == "clear_winner"

    def test_futility(self):
        """Test futility detection."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 500)
        treatment = np.random.normal(100.5, 15, 500)  # Tiny difference

        result = self.stopper.check_stopping_criteria(control, treatment)

        # Should detect futility
        if result["should_stop"]:
            assert result["reason"] == "futility"

    def test_check_futility_method(self):
        """Test futility check method."""
        control = np.array([100] * 100)
        treatment = np.array([100.1] * 100)

        is_futile = self.stopper.check_futility(
            control, treatment, target_mde=0.10
        )

        assert is_futile is True
