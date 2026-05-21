"""Tests for power visualizer."""

from bufferiq.ml.experiments.power.visualizer import PowerVisualizer


class TestPowerVisualizer:
    """Test PowerVisualizer."""

    def setup_method(self):
        """Setup test."""
        self.visualizer = PowerVisualizer()

    def test_create_power_curve(self):
        """Test power curve creation."""
        curve = self.visualizer.create_power_curve(
            baseline_rate=0.05,
            sample_sizes=[1000, 5000, 10000, 20000],
            mde=0.10,
        )

        assert "sample_sizes" in curve
        assert "power_values" in curve
        assert len(curve["sample_sizes"]) == 4
        assert len(curve["power_values"]) == 4

    def test_create_mde_curve(self):
        """Test MDE curve creation."""
        curve = self.visualizer.create_mde_curve(
            baseline_rate=0.05,
            sample_size=10000,
            mde_range=[0.05, 0.10, 0.15, 0.20],
        )

        assert "mde_values" in curve
        assert "power_values" in curve
        assert len(curve["mde_values"]) == 4
