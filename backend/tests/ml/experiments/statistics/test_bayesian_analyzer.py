"""Tests for Bayesian analyzer."""

from bufferiq.ml.experiments.statistics.bayesian_analyzer import BayesianAnalyzer


class TestBayesianAnalyzer:
    """Test BayesianAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = BayesianAnalyzer()

    def test_analyze(self):
        """Test Bayesian analysis."""
        result = self.analyzer.analyze(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=60,
            treatment_total=1000,
        )

        assert 0 <= result.probability_beat_control <= 1
        assert result.expected_loss >= 0
        assert result.posterior_alpha_control > result.prior_alpha
        assert result.posterior_beta_control > result.prior_beta

    def test_calculate_credible_interval(self):
        """Test credible interval calculation."""
        ci_lower, ci_upper = self.analyzer.calculate_credible_interval(
            alpha=51, beta=951, credible_mass=0.95
        )

        assert 0 <= ci_lower <= ci_upper <= 1
