"""Tests for epsilon-greedy."""

from bufferiq.ml.experiments.bandits.epsilon_greedy import EpsilonGreedy
from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class TestEpsilonGreedy:
    """Test EpsilonGreedy."""

    def setup_method(self):
        """Setup test."""
        self.eg = EpsilonGreedy(epsilon=0.1)
        self.arms = [
            BanditArm("control", "Control"),
            BanditArm("treatment", "Treatment"),
        ]

    def test_select_arm(self):
        """Test arm selection."""
        selected = self.eg.select_arm(self.arms)

        assert selected in self.arms

    def test_exploitation(self):
        """Test exploitation of best arm."""
        # Make one arm clearly better
        for _ in range(100):
            self.eg.update(self.arms[0], reward=0.0)
            self.eg.update(self.arms[1], reward=1.0)

        # With epsilon=0, should always pick best
        self.eg.set_epsilon(0.0)
        selected = self.eg.select_arm(self.arms)

        assert selected.variant_id == "treatment"

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.eg.epsilon

        self.eg.decay_epsilon(decay_rate=0.9)

        assert self.eg.epsilon < initial_epsilon
        assert self.eg.epsilon >= 0.01  # Minimum bound
