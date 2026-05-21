"""Tests for report visualizer."""

from bufferiq.ml.experiments.reporting.visualizer import ReportVisualizer


class TestReportVisualizer:
    """Test ReportVisualizer."""

    def setup_method(self):
        """Setup test."""
        self.visualizer = ReportVisualizer()

    def test_create_comparison_chart(self):
        """Test comparison chart creation."""
        chart = self.visualizer.create_comparison_chart(
            control_mean=0.05,
            treatment_mean=0.06,
            ci_lower=0.005,
            ci_upper=0.015,
        )

        assert chart["type"] == "bar"
        assert "data" in chart
        assert "confidence_interval" in chart

    def test_create_time_series_chart(self):
        """Test time series chart creation."""
        chart = self.visualizer.create_time_series_chart(
            dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            control_values=[0.05, 0.05, 0.05],
            treatment_values=[0.06, 0.06, 0.06],
        )

        assert chart["type"] == "line"
        assert len(chart["data"]["datasets"]) == 2

    def test_create_funnel_chart(self):
        """Test funnel chart creation."""
        chart = self.visualizer.create_funnel_chart(
            steps=["View", "Click", "Convert"],
            conversion_rates=[1.0, 0.5, 0.1],
        )

        assert chart["type"] == "funnel"
