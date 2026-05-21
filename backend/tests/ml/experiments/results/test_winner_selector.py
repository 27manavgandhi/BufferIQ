"""Tests for winner selector."""

from bufferiq.ml.experiments.results.winner_selector import WinnerSelector
from bufferiq.ml.experiments.statistics.hypothesis_tester import HypothesisTestResult


class TestWinnerSelector:
    """Test WinnerSelector."""

    def setup_method(self):
        """Setup test."""
        self.selector = WinnerSelector()

    def test_select_winner_significant(self):
        """Test winner selection with significant result."""
        stat_result = HypothesisTestResult(
            test_type="t-test",
            statistic=3.5,
            p_value=0.001,
            is_significant=True,
            alpha=0.05,
            effect_size=0.5,
            effect_size_type="cohen_d",
            ci_lower=0.02,
            ci_upper=0.08,
            confidence_level=0.95,
            n_control=1000,
            n_treatment=1000,
            control_mean=0.05,
            treatment_mean=0.07,
            absolute_diff=0.02,
            relative_diff=0.40,
        )

        result = self.selector.select_winner(stat_result, min_improvement=0.01)

        assert result["has_winner"] is True
        assert result["winner"] == "treatment"
        assert result["meets_minimum"] is True

    def test_select_winner_not_significant(self):
        """Test no winner when not significant."""
        stat_result = HypothesisTestResult(
            test_type="t-test",
            statistic=1.0,
            p_value=0.30,
            is_significant=False,
            alpha=0.05,
            effect_size=0.1,
            effect_size_type="cohen_d",
            ci_lower=-0.01,
            ci_upper=0.03,
            confidence_level=0.95,
            n_control=1000,
            n_treatment=1000,
            control_mean=0.05,
            treatment_mean=0.051,
            absolute_diff=0.001,
            relative_diff=0.02,
        )

        result = self.selector.select_winner(stat_result)

        assert result["has_winner"] is False

    def test_select_winner_bayesian(self):
        """Test Bayesian winner selection."""
        result = self.selector.select_winner_bayesian(
            probability_beat_control=0.96,
            expected_loss=0.005,
            threshold=0.95,
        )

        assert result["has_winner"] is True
        assert result["winner"] == "treatment"
