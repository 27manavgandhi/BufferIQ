"""Tests for group sequential testing."""

import numpy as np
from bufferiq.ml.experiments.sequential.group_sequential import GroupSequentialTester


class TestGroupSequentialTester:
    """Test GroupSequentialTester."""

    def setup_method(self):
        """Setup test."""
        self.tester = GroupSequentialTester(num_looks=3, alpha=0.05)

    def test_boundaries_calculation(self):
        """Test boundary calculation."""
        assert len(self.tester.boundaries) == 3
        # O'Brien-Fleming: boundaries should decrease
        assert self.tester.boundaries[0] > self.tester.boundaries[1]
        assert self.tester.boundaries[1] > self.tester.boundaries[2]

    def test_first_look(self):
        """Test first interim look."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 500)
        treatment = np.random.normal(110, 15, 500)

        result = self.tester.test_at_look(control, treatment, look_number=1)

        assert result["look_number"] == 1
        assert "z_statistic" in result
        assert "z_critical" in result

    def test_invalid_look_number(self):
        """Test invalid look number."""
        import pytest

        control = np.random.normal(100, 15, 100)
        treatment = np.random.normal(110, 15, 100)

        with pytest.raises(ValueError):
            self.tester.test_at_look(control, treatment, look_number=5)
