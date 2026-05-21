"""Tests for bucketing."""

from bufferiq.ml.experiments.assignment.bucketing import HashBucketing
from bufferiq.ml.experiments.design.designer import Variant


class TestHashBucketing:
    """Test HashBucketing."""

    def setup_method(self):
        """Setup test."""
        self.bucketing = HashBucketing()
        self.variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]

    def test_assign_variant_consistency(self):
        """Test variant assignment consistency."""
        variant1 = self.bucketing.assign_variant(
            "exp_001", "user123", self.variants
        )

        variant2 = self.bucketing.assign_variant(
            "exp_001", "user123", self.variants
        )

        assert variant1.id == variant2.id

    def test_assign_variant_distribution(self):
        """Test variant distribution."""
        assignments = {}
        for i in range(1000):
            variant = self.bucketing.assign_variant(
                "exp_001", f"user{i}", self.variants
            )
            assignments[variant.id] = assignments.get(variant.id, 0) + 1

        assert 450 < assignments["control"] < 550
        assert 450 < assignments["treatment"] < 550

    def test_get_bucket(self):
        """Test bucket calculation."""
        bucket = self.bucketing.get_bucket("exp_001", "user123")

        assert 0 <= bucket <= 1