"""Tests for Thompson Sampling."""

from bufferiq.ml.experiments.bandits.thompson_sampling import (
    ThompsonSampling,
    BanditArm,
)


class TestThompsonSampling:
    """Test ThompsonSampling."""

    def setup_method(self):
        """Setup test."""
        self.ts = ThompsonSampling()
        self.arms = [
            BanditArm("control", "Control"),
            BanditArm("treatment_a", "Treatment A"),
            BanditArm("treatment_b", "Treatment B"),
        ]

    def test_select_arm(self):
        """Test arm selection."""
        selected = self.ts.select_arm(self.arms)

        assert selected in self.arms

    def test_update_arm_success(self):
        """Test updating arm with success."""
        arm = self.arms[0]
        initial_alpha = arm.alpha

        self.ts.update(arm, reward=1.0)

        assert arm.trials == 1
        assert arm.successes == 1
        assert arm.alpha == initial_alpha + 1

    def test_update_arm_failure(self):
        """Test updating arm with failure."""
        arm = self.arms[0]
        initial_beta = arm.beta

        self.ts.update(arm, reward=0.0)

        assert arm.trials == 1
        assert arm.successes == 0
        assert arm.beta == initial_beta + 1

    def test_get_arm_statistics(self):
        """Test getting arm statistics."""
        arm = self.arms[0]
        for i in range(10):
            self.ts.update(arm, reward=float(i % 2))

        stats = self.ts.get_arm_statistics(arm)

        assert "expected_value" in stats
        assert "variance" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats

    def test_calculate_probability_best(self):
        """Test probability calculation."""
        # Give one arm better history
        for _ in range(10):
            self.ts.update(self.arms[1], reward=1.0)

        probs = self.ts.calculate_probability_best(self.arms)

        assert len(probs) == 3
        assert sum(probs.values()) > 0.99  # Should sum to ~1
        assert probs["treatment_a"] > probs["control"]
