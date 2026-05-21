"""Tests for summary builder."""

import numpy as np
from bufferiq.ml.experiments.reporting.summary_builder import SummaryBuilder
from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer
from bufferiq.ml.experiments.design.designer import MetricType


class TestSummaryBuilder:
    """Test SummaryBuilder."""

    def setup_method(self):
        """Setup test."""
        self.builder = SummaryBuilder()

    def test_build_executive_summary_launch(self):
        """Test executive summary for launch."""
        analyzer = ResultAnalyzer()
        np.random.seed(42)
        result = analyzer.analyze_experiment(
            control_data=np.random.binomial(1, 0.05, 1000),
            treatment_data=np.random.binomial(1, 0.07, 1000),
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        summary = self.builder.build_executive_summary(result)

        assert "LAUNCH" in summary or "launch" in summary.lower()

    def test_build_insights(self):
        """Test insights building."""
        analyzer = ResultAnalyzer()
        np.random.seed(42)
        result = analyzer.analyze_experiment(
            control_data=np.random.binomial(1, 0.05, 1000),
            treatment_data=np.random.binomial(1, 0.06, 1000),
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        insights = self.builder.build_insights(result)

        assert isinstance(insights, list)
        assert len(insights) > 0
