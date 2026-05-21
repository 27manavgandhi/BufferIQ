"""Tests for hypothesis tester."""

import numpy as np
from bufferiq.ml.experiments.statistics.hypothesis_tester import StatisticalAnalyzer
from bufferiq.ml.experiments.design.designer import MetricType


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = StatisticalAnalyzer()

    def test_continuous_test(self):
        """Test continuous metric analysis."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(110, 15, 1000)

        result = self.analyzer.analyze(
            control_data=control,
            treatment_data=treatment,
            metric_type=MetricType.TIME_ON_PAGE,
        )

        assert result.test_type == "t-test"
        assert result.is_significant is True
        assert result.treatment_mean > result.control_mean

    def test_proportion_test(self):
        """Test proportion analysis."""
        np.random.seed(42)
        control = np.random.binomial(1, 0.05, 1000)
        treatment = np.random.binomial(1, 0.06, 1000)

        result = self.analyzer.analyze(
            control_data=control,
            treatment_data=treatment,
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        assert result.test_type == "z-test"
        assert result.effect_size_type == "cohen_h"

    def test_bayesian_analysis(self):
        """Test Bayesian analysis."""
        result = self.analyzer.bayesian_analyze(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=60,
            treatment_total=1000,
        )

        assert 0 <= result.probability_beat_control <= 1
        assert result.expected_loss >= 0

    def test_mann_whitney(self):
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        control = np.random.exponential(10, 100)
        treatment = np.random.exponential(12, 100)

        result = self.analyzer._mann_whitney_test(
            control, treatment, alpha=0.05, confidence_level=0.95
        )

        assert result.test_type == "mann-whitney"
        assert result.effect_size_type == "cliff_delta"
