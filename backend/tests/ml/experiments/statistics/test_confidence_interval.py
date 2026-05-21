"""Tests for confidence interval calculator."""

import numpy as np
from bufferiq.ml.experiments.statistics.confidence_interval import ConfidenceIntervalCalculator


class TestConfidenceIntervalCalculator:
    """Test ConfidenceIntervalCalculator."""

    def setup_method(self):
        """Setup test."""
        self.calculator = ConfidenceIntervalCalculator()

    def test_mean_difference_ci(self):
        """Test mean difference CI."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(110, 15, 1000)

        ci_lower, ci_upper = self.calculator.mean_difference_ci(
            control, treatment, confidence_level=0.95
        )

        mean_diff = np.mean(treatment) - np.mean(control)

        # CI should contain mean difference
        assert ci_lower < mean_diff < ci_upper

    def test_proportion_difference_ci(self):
        """Test proportion difference CI."""
        ci_lower, ci_upper = self.calculator.proportion_difference_ci(
            p_control=0.05,
            p_treatment=0.06,
            n_control=1000,
            n_treatment=1000,
            confidence_level=0.95,
        )

        diff = 0.06 - 0.05

        # CI should contain difference
        assert ci_lower < diff < ci_upper
