"""Tests for sample size calculator."""

import pytest
from bufferiq.ml.experiments.design.sample_size_calculator import SampleSizeCalculator


class TestSampleSizeCalculator:
    """Test SampleSizeCalculator."""

    def setup_method(self):
        """Setup test."""
        self.calculator = SampleSizeCalculator()

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        n = self.calculator.calculate(
            baseline_rate=0.05,
            mde=0.10,
            alpha=0.05,
            power=0.80,
        )

        assert n > 0
        assert isinstance(n, int)
        assert 1000 < n < 100000

    def test_calculate_power(self):
        """Test power calculation."""
        power = self.calculator.calculate_power(
            baseline_rate=0.05,
            treatment_rate=0.055,
            sample_size=10000,
            alpha=0.05,
        )

        assert 0 <= power <= 1

    def test_calculate_mde(self):
        """Test MDE calculation."""
        mde = self.calculator.calculate_mde(
            baseline_rate=0.05,
            sample_size=10000,
            alpha=0.05,
            power=0.80,
        )

        assert mde > 0
        assert mde < 1

    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple variants."""
        n_two = self.calculator.calculate(
            baseline_rate=0.05,
            mde=0.10,
            num_variants=2,
        )

        n_five = self.calculator.calculate(
            baseline_rate=0.05,
            mde=0.10,
            num_variants=5,
        )

        assert n_five > n_two

    def test_invalid_parameters(self):
        """Test invalid parameters."""
        with pytest.raises(ValueError):
            self.calculator.calculate(baseline_rate=1.5, mde=0.10)

        with pytest.raises(ValueError):
            self.calculator.calculate(baseline_rate=0.05, mde=1.5)