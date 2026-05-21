"""Tests for contextual bandit."""

from bufferiq.ml.experiments.bandits.contextual import ContextualBandit
from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class TestContextualBandit:
    """Test ContextualBandit."""

    def setup_method(self):
        """Setup test."""
        self.cb = ContextualBandit(n_features=3)
        self.arms = [
            BanditArm("control", "Control"),
            BanditArm("treatment", "Treatment"),
        ]

    def test_context_to_features(self):
        """Test context conversion."""
        context = {"age": 25, "tenure": 30, "segment": 1}

        features = self.cb._context_to_features(context)

        assert len(features) == 3

    def test_select_arm(self):
        """Test arm selection with context."""
        context = {"age": 25, "tenure": 30, "segment": 1}

        selected = self.cb.select_arm(self.arms, context)

        assert selected in self.arms

    def test_update_with_context(self):
        """Test updating with context."""
        context = {"age": 25, "tenure": 30, "segment": 1}

        self.cb.update(self.arms[0], context, reward=1.0)

        # Model should be initialized
        assert self.arms[0].variant_id in self.cb.arm_models
