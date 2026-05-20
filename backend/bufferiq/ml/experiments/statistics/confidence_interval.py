"""
Confidence interval calculator.

Calculates confidence intervals for various statistics.

Example:
```python
    calc = ConfidenceIntervalCalculator()
    
    ci_lower, ci_upper = calc.mean_difference_ci(
        control, treatment, confidence_level=0.95
    )
```
"""

import numpy as np
from scipy import stats


class ConfidenceIntervalCalculator:
    """
    Calculate confidence intervals.

    Example:
```python
        calc = ConfidenceIntervalCalculator()

        ci_lower, ci_upper = calc.mean_difference_ci(
            control=[100, 105, 98],
            treatment=[110, 115, 108],
            confidence_level=0.95
        )
```
    """

    def mean_difference_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Calculate CI for mean difference.

        Args:
            control: Control data
            treatment: Treatment data
            confidence_level: Confidence level

        Returns:
            (lower, upper) confidence interval
        """
        # Calculate statistics
        mean_diff = float(np.mean(treatment) - np.mean(control))
        n_control = len(control)
        n_treatment = len(treatment)

        # Pooled standard deviation
        var_control = np.var(control, ddof=1)
        var_treatment = np.var(treatment, ddof=1)

        pooled_std = np.sqrt(
            ((n_control - 1) * var_control + (n_treatment - 1) * var_treatment)
            / (n_control + n_treatment - 2)
        )

        # Standard error
        se = pooled_std * np.sqrt(1 / n_control + 1 / n_treatment)

        # Degrees of freedom
        df = n_control + n_treatment - 2

        # Critical value
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Confidence interval
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        return float(ci_lower), float(ci_upper)

    def proportion_difference_ci(
        self,
        p_control: float,
        p_treatment: float,
        n_control: int,
        n_treatment: int,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Calculate CI for proportion difference.

        Args:
            p_control: Control proportion
            p_treatment: Treatment proportion
            n_control: Control sample size
            n_treatment: Treatment sample size
            confidence_level: Confidence level

        Returns:
            (lower, upper) confidence interval
        """
        # Difference
        diff = p_treatment - p_control

        # Standard error
        se = np.sqrt(
            p_control * (1 - p_control) / n_control
            + p_treatment * (1 - p_treatment) / n_treatment
        )

        # Critical value
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Confidence interval
        ci_lower = diff - z_crit * se
        ci_upper = diff + z_crit * se

        return float(ci_lower), float(ci_upper)

    def relative_difference_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """
        Calculate CI for relative difference using bootstrap.

        Args:
            control: Control data
            treatment: Treatment data
            confidence_level: Confidence level
            n_bootstrap: Bootstrap samples

        Returns:
            (lower, upper) confidence interval
        """
        relative_diffs = []

        for _ in range(n_bootstrap):
            # Resample
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(
                treatment, size=len(treatment), replace=True
            )

            # Calculate relative difference
            mean_c = np.mean(control_sample)
            mean_t = np.mean(treatment_sample)

            if mean_c > 0:
                rel_diff = (mean_t - mean_c) / mean_c
                relative_diffs.append(rel_diff)

        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = float(np.percentile(relative_diffs, lower_percentile))
        ci_upper = float(np.percentile(relative_diffs, upper_percentile))

        return ci_lower, ci_upper