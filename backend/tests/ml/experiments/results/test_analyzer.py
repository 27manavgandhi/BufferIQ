"""Tests for result analyzer."""

import numpy as np
from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer
from bufferiq.ml.experiments.design.designer import MetricType


class TestResultAnalyzer:
    """Test ResultAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = ResultAnalyzer()

    def test_analyze_experiment_winner(self):
        """Test experiment analysis with clear winner."""
        np.random.seed(42)
        control = np.random.binomial(1, 0.05, 1000)
        treatment = np.random.binomial(1, 0.07, 1000)

        result = self.analyzer.analyze_experiment(
            control_data=control,
            treatment_data=treatment,
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        assert result.statistical_result.is_significant
        assert result.has_winner

    def test_analyze_experiment_no_winner(self):
        """Test experiment with no clear winner."""
        np.random.seed(42)
        control = np.random.binomial(1, 0.05, 1000)
        treatment = np.random.binomial(1, 0.051, 1000)

        result = self.analyzer.analyze_experiment(
            control_data=control,
            treatment_data=treatment,
            metric_type=MetricType.ENGAGEMENT_RATE,
        )

        assert result.has_winner is False or result.confidence < 0.95

    def test_insufficient_data(self):
        """Test insufficient data handling."""
        control = np.array([1, 0, 1])
        treatment = np.array([1, 1, 0])

        result = self.analyzer.analyze_experiment(
            control_data=control,
            treatment_data=treatment,
            metric_type=MetricType.ENGAGEMENT_RATE,
            min_sample_size=100,
        )

        assert result.recommendation.startswith("Insufficient data")
