"""Tests for interference detector."""

from bufferiq.ml.experiments.interference.detector import InterferenceDetector


class TestInterferenceDetector:
    """Test InterferenceDetector."""

    def setup_method(self):
        """Setup test."""
        self.detector = InterferenceDetector()

    def test_detect_interference(self):
        """Test interference detection."""
        result = self.detector.detect_interference(
            treatment_user_ids=["u1", "u2"],
            control_user_ids=["u3", "u4"],
            treatment_outcomes=[1.0, 1.0],
            control_outcomes=[0.8, 0.5],
            network_edges=[("u1", "u3"), ("u2", "u4")],
        )

        assert "has_interference" in result
        assert "cross_edges_count" in result

    def test_calculate_exposure_probability(self):
        """Test exposure probability calculation."""
        prob = self.detector.calculate_exposure_probability(
            user_id="u3",
            treatment_user_ids=["u1", "u2"],
            network_edges=[("u1", "u3"), ("u2", "u3"), ("u3", "u4")],
        )

        assert 0 <= prob <= 1
        assert prob == 2 / 3  # 2 out of 3 neighbors are treatment

    def test_detect_spillover(self):
        """Test spillover detection."""
        result = self.detector.detect_spillover(
            control_outcomes_near_treatment=[0.7, 0.8, 0.75],
            control_outcomes_far_from_treatment=[0.5, 0.5, 0.5],
        )

        assert "has_spillover" in result
        assert "spillover_effect" in result
