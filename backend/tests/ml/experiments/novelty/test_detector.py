"""Tests for novelty detector."""

from bufferiq.ml.experiments.novelty.detector import NoveltyDetector


class TestNoveltyDetector:
    """Test NoveltyDetector."""

    def setup_method(self):
        """Setup test."""
        self.detector = NoveltyDetector()

    def test_detect_novelty_effect(self):
        """Test novelty effect detection."""
        # Declining treatment effect
        treatment_means = [0.10, 0.09, 0.08, 0.07, 0.06]
        control_means = [0.05, 0.05, 0.05, 0.05, 0.05]

        result = self.detector.detect_novelty(treatment_means, control_means)

        assert result["has_novelty_effect"] is True
        assert result["decay_rate"] < 0

    def test_no_novelty_effect(self):
        """Test no novelty effect."""
        # Stable treatment effect
        treatment_means = [0.10, 0.10, 0.10, 0.10, 0.10]
        control_means = [0.05, 0.05, 0.05, 0.05, 0.05]

        result = self.detector.detect_novelty(treatment_means, control_means)

        assert result["has_novelty_effect"] is False

    def test_insufficient_data(self):
        """Test insufficient data handling."""
        treatment_means = [0.10, 0.09]
        control_means = [0.05, 0.05]

        result = self.detector.detect_novelty(treatment_means, control_means)

        assert result["has_novelty_effect"] is False
        assert result["reason"] == "insufficient_data"

    def test_estimate_stabilization_time(self):
        """Test stabilization time estimation."""
        treatment_means = [0.10, 0.09, 0.08, 0.07, 0.07, 0.07, 0.07]
        control_means = [0.05] * 7

        days = self.detector.estimate_stabilization_time(
            treatment_means, control_means
        )

        assert days >= 0
