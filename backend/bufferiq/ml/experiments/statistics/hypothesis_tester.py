"""
Hypothesis testing.

Performs statistical hypothesis tests including t-tests, z-tests,
Mann-Whitney U, chi-square, and Bayesian analysis.

Key features:
    - T-test for continuous metrics
    - Proportion test for binary metrics
    - Mann-Whitney U test
    - Chi-square test
    - Effect size calculation
    - Confidence intervals
    - Bayesian analysis

Example:
```python
    analyzer = StatisticalAnalyzer()
    
    result = analyzer.analyze(
        control_data=control_values,
        treatment_data=treatment_values,
        metric_type=MetricType.ENGAGEMENT_RATE,
        alpha=0.05
    )
    
    if result.is_significant:
        print(f"Treatment wins! p={result.p_value:.4f}")
```
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from bufferiq.ml.experiments.design.designer import MetricType
from bufferiq.ml.experiments.statistics.effect_size import EffectSizeCalculator
from bufferiq.ml.experiments.statistics.confidence_interval import (
    ConfidenceIntervalCalculator,
)


@dataclass
class HypothesisTestResult:
    """Hypothesis test result."""

    test_type: str  # "t-test", "z-test", "mann-whitney", "chi-square"
    statistic: float
    p_value: float

    # Interpretation
    is_significant: bool  # p < alpha
    alpha: float

    # Effect size
    effect_size: float
    effect_size_type: str  # "cohen_d", "hedge_g", "cliff_delta"

    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Sample sizes
    n_control: int
    n_treatment: int

    # Means/rates
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff: float  # % change


@dataclass
class BayesianResult:
    """Bayesian analysis result."""

    probability_beat_control: float  # P(treatment > control)
    expected_loss: float
    credible_interval_lower: float
    credible_interval_upper: float

    # Prior parameters
    prior_alpha: float
    prior_beta: float

    # Posterior parameters
    posterior_alpha_control: float
    posterior_beta_control: float
    posterior_alpha_treatment: float
    posterior_beta_treatment: float


class StatisticalAnalyzer:
    """
    Perform statistical hypothesis testing.

    Supports t-tests, z-tests, Mann-Whitney U, chi-square,
    and Bayesian analysis for A/B tests.

    Example:
```python
        analyzer = StatisticalAnalyzer()

        result = analyzer.analyze(
            control_data=[100, 105, 98, 102, ...],
            treatment_data=[110, 115, 108, 112, ...],
            metric_type=MetricType.ENGAGEMENT_RATE,
            alpha=0.05
        )

        if result.is_significant:
            print(f"✓ Treatment is significantly better!")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Effect size: {result.effect_size:.2f}")
            print(f"  Relative lift: {result.relative_diff:.1%}")
        else:
            print(f"✗ No significant difference detected")
```
    """

    def __init__(self) -> None:
        """Initialize statistical analyzer."""
        self.effect_size_calc = EffectSizeCalculator()
        self.ci_calc = ConfidenceIntervalCalculator()

    def analyze(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        metric_type: MetricType,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        test_type: Optional[str] = None,
    ) -> HypothesisTestResult:
        """
        Perform hypothesis test.

        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            metric_type: Type of metric
            alpha: Significance level
            confidence_level: Confidence level for CI
            test_type: Optional specific test type

        Returns:
            Test result with statistics
        """
        # Convert to numpy arrays
        control = np.asarray(control_data)
        treatment = np.asarray(treatment_data)

        # Choose test based on metric type
        if test_type:
            if test_type == "t-test":
                return self._continuous_test(
                    control, treatment, alpha, confidence_level
                )
            elif test_type == "mann-whitney":
                return self._mann_whitney_test(
                    control, treatment, alpha, confidence_level
                )
            elif test_type == "proportion":
                return self._proportion_test(
                    control, treatment, alpha, confidence_level
                )
            else:
                raise ValueError(f"Unknown test type: {test_type}")

        # Auto-select test
        if metric_type in [
            MetricType.ENGAGEMENT_RATE,
            MetricType.CONVERSION_RATE,
            MetricType.CLICK_THROUGH_RATE,
        ]:
            return self._proportion_test(control, treatment, alpha, confidence_level)
        else:
            return self._continuous_test(control, treatment, alpha, confidence_level)

    def _continuous_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float,
        confidence_level: float,
    ) -> HypothesisTestResult:
        """
        Perform t-test for continuous metrics.

        Args:
            control: Control data
            treatment: Treatment data
            alpha: Significance level
            confidence_level: Confidence level

        Returns:
            Test result
        """
        # Two-sample t-test
        statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        # Effect size (Cohen's d)
        cohen_d = self.effect_size_calc.cohens_d(control, treatment)

        # Confidence interval
        mean_diff = float(np.mean(treatment) - np.mean(control))
        ci_lower, ci_upper = self.ci_calc.mean_difference_ci(
            control, treatment, confidence_level
        )

        # Calculate relative difference
        control_mean = float(np.mean(control))
        treatment_mean = float(np.mean(treatment))
        relative_diff = mean_diff / control_mean if control_mean != 0 else 0.0

        return HypothesisTestResult(
            test_type="t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            alpha=alpha,
            effect_size=cohen_d,
            effect_size_type="cohen_d",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_control=len(control),
            n_treatment=len(treatment),
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_diff=mean_diff,
            relative_diff=relative_diff,
        )

    def _proportion_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float,
        confidence_level: float,
    ) -> HypothesisTestResult:
        """
        Perform proportion test (z-test) for binary metrics.

        Args:
            control: Control data (0/1)
            treatment: Treatment data (0/1)
            alpha: Significance level
            confidence_level: Confidence level

        Returns:
            Test result
        """
        # Calculate proportions
        p_control = float(np.mean(control))
        p_treatment = float(np.mean(treatment))

        n_control = len(control)
        n_treatment = len(treatment)

        # Pooled proportion
        p_pooled = (np.sum(control) + np.sum(treatment)) / (n_control + n_treatment)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_control + 1 / n_treatment))

        # Z-statistic
        z_stat = (p_treatment - p_control) / se if se > 0 else 0.0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Effect size (Cohen's h)
        cohen_h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))

        # Confidence interval
        ci_lower, ci_upper = self.ci_calc.proportion_difference_ci(
            p_control, p_treatment, n_control, n_treatment, confidence_level
        )

        # Relative difference
        diff = p_treatment - p_control
        relative_diff = diff / p_control if p_control > 0 else 0.0

        return HypothesisTestResult(
            test_type="z-test",
            statistic=float(z_stat),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            alpha=alpha,
            effect_size=float(cohen_h),
            effect_size_type="cohen_h",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_control=n_control,
            n_treatment=n_treatment,
            control_mean=p_control,
            treatment_mean=p_treatment,
            absolute_diff=diff,
            relative_diff=relative_diff,
        )

    def _mann_whitney_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float,
        confidence_level: float,
    ) -> HypothesisTestResult:
        """
        Perform Mann-Whitney U test (non-parametric).

        Args:
            control: Control data
            treatment: Treatment data
            alpha: Significance level
            confidence_level: Confidence level

        Returns:
            Test result
        """
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            treatment, control, alternative="two-sided"
        )

        # Effect size (Cliff's delta)
        cliff_delta = self.effect_size_calc.cliffs_delta(control, treatment)

        # Medians
        control_median = float(np.median(control))
        treatment_median = float(np.median(treatment))
        diff = treatment_median - control_median

        # CI (bootstrap)
        ci_lower, ci_upper = self._bootstrap_ci(
            control, treatment, confidence_level, statistic_fn=np.median
        )

        relative_diff = diff / control_median if control_median != 0 else 0.0

        return HypothesisTestResult(
            test_type="mann-whitney",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            alpha=alpha,
            effect_size=cliff_delta,
            effect_size_type="cliff_delta",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_control=len(control),
            n_treatment=len(treatment),
            control_mean=control_median,
            treatment_mean=treatment_median,
            absolute_diff=diff,
            relative_diff=relative_diff,
        )

    def _bootstrap_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        confidence_level: float,
        statistic_fn: callable = np.mean,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            control: Control data
            treatment: Treatment data
            confidence_level: Confidence level
            statistic_fn: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples

        Returns:
            (lower, upper) confidence interval
        """
        diffs = []

        for _ in range(n_bootstrap):
            # Resample
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(
                treatment, size=len(treatment), replace=True
            )

            # Calculate difference
            diff = statistic_fn(treatment_sample) - statistic_fn(control_sample)
            diffs.append(diff)

        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = float(np.percentile(diffs, lower_percentile))
        ci_upper = float(np.percentile(diffs, upper_percentile))

        return ci_lower, ci_upper

    def bayesian_analyze(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100000,
    ) -> BayesianResult:
        """
        Perform Bayesian analysis for proportions.

        Args:
            control_conversions: Control successes
            control_total: Control sample size
            treatment_conversions: Treatment successes
            treatment_total: Treatment sample size
            prior_alpha: Beta prior alpha
            prior_beta: Beta prior beta
            n_samples: Monte Carlo samples

        Returns:
            Bayesian analysis result
        """
        # Posterior parameters
        post_alpha_c = prior_alpha + control_conversions
        post_beta_c = prior_beta + (control_total - control_conversions)
        post_alpha_t = prior_alpha + treatment_conversions
        post_beta_t = prior_beta + (treatment_total - treatment_conversions)

        # Monte Carlo simulation
        control_samples = np.random.beta(post_alpha_c, post_beta_c, n_samples)
        treatment_samples = np.random.beta(post_alpha_t, post_beta_t, n_samples)

        # Probability treatment > control
        prob_beat = float(np.mean(treatment_samples > control_samples))

        # Expected loss
        loss = np.maximum(control_samples - treatment_samples, 0)
        expected_loss = float(np.mean(loss))

        # Credible interval
        diff_samples = treatment_samples - control_samples
        ci_lower, ci_upper = np.percentile(diff_samples, [2.5, 97.5])

        return BayesianResult(
            probability_beat_control=prob_beat,
            expected_loss=expected_loss,
            credible_interval_lower=float(ci_lower),
            credible_interval_upper=float(ci_upper),
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            posterior_alpha_control=post_alpha_c,
            posterior_beta_control=post_beta_c,
            posterior_alpha_treatment=post_alpha_t,
            posterior_beta_treatment=post_beta_t,
        )