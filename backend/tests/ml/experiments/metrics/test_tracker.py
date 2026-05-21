"""Tests for metrics tracker."""

from unittest.mock import Mock
from bufferiq.ml.experiments.metrics.tracker import MetricsTracker
from bufferiq.ml.experiments.design.designer import MetricType


class TestMetricsTracker:
    """Test MetricsTracker."""

    def setup_method(self):
        """Setup test."""
        self.db = Mock()
        self.tracker = MetricsTracker(self.db)

    def test_track_metric(self):
        """Test tracking metric."""
        self.tracker.track(
            experiment_id="exp_001",
            user_id="user123",
            variant_id="treatment",
            metric_type=MetricType.ENGAGEMENT_RATE,
            value=1.0,
        )

        metrics = self.tracker.get_metrics("exp_001")
        assert len(metrics) == 1
        assert metrics[0].value == 1.0

    def test_get_metric_values(self):
        """Test getting metric values."""
        for i in range(10):
            self.tracker.track(
                experiment_id="exp_001",
                user_id=f"user{i}",
                variant_id="treatment",
                metric_type=MetricType.ENGAGEMENT_RATE,
                value=float(i),
            )

        values = self.tracker.get_metric_values(
            experiment_id="exp_001",
            variant_id="treatment",
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        assert len(values) == 10
        assert sum(values) == 45  # 0+1+2+...+9
