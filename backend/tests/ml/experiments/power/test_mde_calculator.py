"""Tests for MDE calculator."""

from bufferiq.ml.experiments.power.mde_calculator import MDECalculator


class TestMDECalculator:
    """Test MDECalculator."""

    def setup_method(self):
        """Setup test."""
        self.calculator = MDECalculator()

    def test_calculate_mde(self):
        """Test MDE calculation."""
        mde = self.calculator.calculate(
            baseline_rate=0.05,
            sample_size=10000,
            power=0.80,
            alpha=0.05,
        )

        assert mde > 0
        assert mde < 1

    def test_mde_decreases_with_sample_size(self):
        """Test MDE decreases with larger sample."""
        mde_small = self.calculator.calculate(
            baseline_rate=0.05, sample_size=1000
        )

        mde_large = self.calculator.calculate(
            baseline_rate=0.05, sample_size=10000
        )

        assert mde_large < mde_small
