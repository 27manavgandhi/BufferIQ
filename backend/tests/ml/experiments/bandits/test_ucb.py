"""Tests for UCB."""

from bufferiq.ml.experiments.bandits.ucb import UCB
from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class TestUCB:
    """Test UCB."""

    def setup_method(self):
        """Setup test."""
        self.ucb = UCB(exploration_param=2.0)
        self.arms = [
            BanditArm("control", "Control"),
            BanditArm("treatment", "Treatment"),
        ]

    def test_select_untried_arm(self):
        """Test selecting untried arm first."""
        selected = self.ucb.select_arm(self.arms, total_trials=10)

        # Should select first untried arm
        assert selected.trials == 0

    def test_select_with_history(self):
        """Test selection with history."""
        # Give arms some history
        for _ in range(5):
            self.ucb.update(self.arms[0], reward=1.0)

        for _ in range(10):
            self.ucb.update(self.arms[1], reward=0.0)

        selected = self.ucb.select_arm(self.arms, total_trials=15)

        # Should select based on UCB
        assert selected in self.arms

    def test_get_ucb_value(self):
        """Test UCB value calculation."""
        self.ucb.update(self.arms[0], reward=1.0)

        ucb_value = self.ucb.get_ucb_value(self.arms[0], total_trials=10)

        assert ucb_value > 0
