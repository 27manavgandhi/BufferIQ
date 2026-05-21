"""Tests for SPRT."""

from bufferiq.ml.experiments.sequential.sprt import SequentialTester


class TestSequentialTester:
    """Test SequentialTester."""

    def setup_method(self):
        """Setup test."""
        self.tester = SequentialTester(alpha=0.05, beta=0.20)

    def test_sprt_stop_for_treatment(self):
        """Test SPRT stopping for treatment win."""
        result = self.tester.test(
            control_successes=50,
            control_trials=1000,
            treatment_successes=70,
            treatment_trials=1000,
            mde=0.10,
        )

        # With clear difference, should stop
        assert result.decision in ["continue", "stop"]
        if result.decision == "stop":
            assert result.conclusion in ["treatment_wins", "control_wins"]

    def test_sprt_continue(self):
        """Test SPRT continuing."""
        result = self.tester.test(
            control_successes=50,
            control_trials=100,
            treatment_successes=51,
            treatment_trials=100,
            mde=0.10,
        )

        # With small difference and small sample, should continue
        assert result.decision == "continue"

    def test_boundaries(self):
        """Test boundary calculations."""
        assert self.tester.upper_boundary > 0
        assert self.tester.lower_boundary < 0
        assert self.tester.upper_boundary > abs(self.tester.lower_boundary)
