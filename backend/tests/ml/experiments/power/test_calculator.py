"""Tests for power analyzer."""

from bufferiq.ml.experiments.power.calculator import PowerAnalyzer


class TestPowerAnalyzer:
    """Test PowerAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = PowerAnalyzer()

    def test_calculate_power(self):
        """Test power calculation."""
        power = self.analyzer.calculate_power(
            baseline_rate=0.05,
            treatment_rate=0.055,
            sample_size=10000,
            alpha=0.05,
        )

        assert 0 <= power <= 1

    def test_calculate_required_sample_size(self):
        """Test sample size calculation."""
        n = self.analyzer.calculate_required_sample_size(
            baseline_rate=0.05,
            treatment_rate=0.055,
            power=0.80,
            alpha=0.05,
        )

        assert n > 0
        assert isinstance(n, int)

    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        power_small = self.analyzer.calculate_power(
            baseline_rate=0.05,
            treatment_rate=0.055,
            sample_size=1000,
        )

        power_large = self.analyzer.calculate_power(
            baseline_rate=0.05,
            treatment_rate=0.055,
            sample_size=10000,
        )

        assert power_large > power_small
