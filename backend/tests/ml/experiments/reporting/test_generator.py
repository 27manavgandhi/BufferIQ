"""Tests for report generator."""

import numpy as np
from bufferiq.ml.experiments.reporting.generator import ReportGenerator
from bufferiq.ml.experiments.design.designer import (
    ExperimentDesigner,
    Variant,
    MetricType,
)
from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer


class TestReportGenerator:
    """Test ReportGenerator."""

    def setup_method(self):
        """Setup test."""
        self.generator = ReportGenerator()

    def test_generate_report(self):
        """Test report generation."""
        # Create experiment config
        designer = ExperimentDesigner()
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]
        config = designer.design(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        # Create result
        analyzer = ResultAnalyzer()
        np.random.seed(42)
        result = analyzer.analyze_experiment(
            control_data=np.random.binomial(1, 0.05, 1000),
            treatment_data=np.random.binomial(1, 0.06, 1000),
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        # Generate report
        report = self.generator.generate_report(config, result)

        assert report.experiment_id == config.experiment_id
        assert report.experiment_name == config.name
        assert len(report.recommendations) > 0
        assert len(report.next_steps) > 0

    def test_export_to_markdown(self):
        """Test markdown export."""
        designer = ExperimentDesigner()
        variants = [
            Variant("control", "Control", "Original", 0.5, {}, True),
            Variant("treatment", "Treatment", "New", 0.5, {}),
        ]
        config = designer.design(
            name="Test",
            description="Test",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
        )

        analyzer = ResultAnalyzer()
        np.random.seed(42)
        result = analyzer.analyze_experiment(
            control_data=np.random.binomial(1, 0.05, 1000),
            treatment_data=np.random.binomial(1, 0.06, 1000),
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        report = self.generator.generate_report(config, result)
        markdown = self.generator.export_to_markdown(report)

        assert "# Experiment Report" in markdown
        assert config.name in markdown
