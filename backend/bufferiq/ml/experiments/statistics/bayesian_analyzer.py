"""
Bayesian analyzer.

Performs Bayesian inference for A/B tests.

Example:
```python
    analyzer = BayesianAnalyzer()
    
    result = analyzer.analyze(
        control_conversions=50,
        control_total=1000,
        treatment_conversions=60,
        treatment_total=1000
    )
    
    print(f"P(treatment > control): {result.probability_beat_control:.1%}")
```
"""

import numpy as np
from scipy import stats

from bufferiq.ml.experiments.statistics.hypothesis_tester import BayesianResult


class BayesianAnalyzer:
    """
    Bayesian inference for A/B tests.

    Example:
```python
        analyzer = BayesianAnalyzer()

        result = analyzer.analyze(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=60,
            treatment_total=1000
        )

        if result.probability_beat_control > 0.95:
            print("Treatment wins with 95% confidence")
```
    """

    def analyze(
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
        Perform Bayesian analysis.

        Args:
            control_conversions: Control successes
            control_total: Control total
            treatment_conversions: Treatment successes
            treatment_total: Treatment total
            prior_alpha: Prior alpha
            prior_beta: Prior beta
            n_samples: MC samples

        Returns:
            Bayesian result
        """
        # Posterior parameters
        post_alpha_c = prior_alpha + control_conversions
        post_beta_c = prior_beta + (control_total - control_conversions)
        post_alpha_t = prior_alpha + treatment_conversions
        post_beta_t = prior_beta + (treatment_total - treatment_conversions)

        # Sample from posteriors
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

    def calculate_credible_interval(
        self,
        alpha: float,
        beta: float,
        credible_mass: float = 0.95,
        n_samples: int = 100000,
    ) -> tuple[float, float]:
        """
        Calculate credible interval for Beta distribution.

        Args:
            alpha: Beta alpha parameter
            beta: Beta beta parameter
            credible_mass: Credible mass
            n_samples: Number of samples

        Returns:
            (lower, upper) credible interval
        """
        samples = np.random.beta(alpha, beta, n_samples)

        alpha_level = 1 - credible_mass
        lower_percentile = (alpha_level / 2) * 100
        upper_percentile = (1 - alpha_level / 2) * 100

        ci_lower = float(np.percentile(samples, lower_percentile))
        ci_upper = float(np.percentile(samples, upper_percentile))

        return ci_lower, ci_upper