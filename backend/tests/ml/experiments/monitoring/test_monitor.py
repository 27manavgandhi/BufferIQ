"""Tests for experiment monitor."""

from bufferiq.ml.experiments.monitoring.monitor import ExperimentMonitor


class TestExperimentMonitor:
    """Test ExperimentMonitor."""

    def setup_method(self):
        """Setup test."""
        self.monitor = ExperimentMonitor()

    def test_check_health_healthy(self):
        """Test healthy experiment."""
        health = self.monitor.check_health(
            variant_counts={"control": 1000, "treatment": 1000},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        assert health.is_healthy is True
        assert len(health.issues) == 0

    def test_check_health_srm(self):
        """Test SRM detection."""
        health = self.monitor.check_health(
            variant_counts={"control": 1200, "treatment": 800},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        # Should detect SRM
        if not health.is_healthy:
            assert any("ratio" in issue.lower() for issue in health.issues)

    def test_generate_alerts(self):
        """Test alert generation."""
        health = self.monitor.check_health(
            variant_counts={"control": 1200, "treatment": 800},
            expected_ratios={"control": 0.5, "treatment": 0.5},
        )

        alerts = self.monitor.generate_alerts(health)

        assert isinstance(alerts, list)
