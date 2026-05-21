"""Tests for randomizer."""

from bufferiq.ml.experiments.design.randomizer import Randomizer
from bufferiq.ml.experiments.design.designer import Variant


class TestRandomizer:
    """Test Randomizer."""

    def setup_method(self):
        """Setup test."""
        self.randomizer = Randomizer(seed=42)
        self.variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

    def test_hash_based_consistency(self):
        """Test hash-based assignment consistency."""
        variant1 = self.randomizer.hash_based(
            self.variants, "user123", "exp_001"
        )

        variant2 = self.randomizer.hash_based(
            self.variants, "user123", "exp_001"
        )

        assert variant1.id == variant2.id

    def test_hash_based_distribution(self):
        """Test hash-based distribution."""
        assignments = {}
        for i in range(1000):
            variant = self.randomizer.hash_based(
                self.variants, f"user{i}", "exp_001"
            )
            assignments[variant.id] = assignments.get(variant.id, 0) + 1

        assert 450 < assignments["control"] < 550
        assert 450 < assignments["treatment"] < 550

    def test_blocked_randomization(self):
        """Test blocked randomization."""
        block = self.randomizer.blocked_random(self.variants, block_size=10)

        assert len(block) == 10

        control_count = sum(1 for v in block if v.id == "control")
        treatment_count = sum(1 for v in block if v.id == "treatment")

        assert control_count == 5
        assert treatment_count == 5