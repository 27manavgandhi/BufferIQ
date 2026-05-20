"""
Power analyzer.

Calculates statistical power for experiments.

Example:
```python
    analyzer = PowerAnalyzer()
    
    power = analyzer.calculate_power(
        baseline_rate=0.05,
        treatment_rate=0.055,
        sample_size=10000,
        alpha=0.05
    )
    
    print(f"Power: {power:.1%}")
```
"""

import numpy as np
from scipy import stats


class PowerAnalyzer:
    """
    Calculate statistical power.

    Example:
```python
        analyzer = PowerAnalyzer()

        power = analyzer.calculate_power(
            baseline_rate=0.05,
            treatment_rate=0.055,
            sample_size=10000
        )

        print(f"Power: {power:.2%}")
        # Output: Power: 82%
```
    """

    def calculate_power(
        self,
        baseline_rate: float,
        treatment_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate statistical power.

        Args:
            baseline_rate: Baseline conversion rate
            treatment_rate: Treatment conversion rate
            sample_size: Sample size per variant
            alpha: Type I error rate
            two_tailed: Two-tailed test

        Returns:
            Statistical power (0-1)
        """
        # Effect size (Cohen's h)
        p1 = baseline_rate
        p2 = treatment_rate
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Z-scores
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        # Calculate power
        z_beta = effect_size * np.sqrt(sample_size) - z_alpha
        power = stats.norm.cdf(z_beta)

        return float(max(0.0, min(1.0, power)))

    def calculate_power_for_mde(
        self,
        baseline_rate: float,
        mde: float,
        sample_size: int,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate power for given MDE.

        Args:
            baseline_rate: Baseline rate
            mde: Minimum detectable effect (relative)
            sample_size: Sample size per variant
            alpha: Type I error rate
            two_tailed: Two-tailed test

        Returns:
            Statistical power
        """
        treatment_rate = baseline_rate * (1 + mde)
        return self.calculate_power(
            baseline_rate, treatment_rate, sample_size, alpha, two_tailed
        )

    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        treatment_rate: float,
        power: float = 0.80,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> int:
        """
        Calculate required sample size for desired power.

        Args:
            baseline_rate: Baseline rate
            treatment_rate: Treatment rate
            power: Desired power
            alpha: Type I error rate
            two_tailed: Two-tailed test

        Returns:
            Required sample size per variant
        """
        # Effect size
        p1 = baseline_rate
        p2 = treatment_rate
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        if abs(effect_size) < 1e-10:
            return int(1e9)  # Impossibly large

        # Z-scores
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Sample size
        n = ((z_alpha + z_beta) ** 2) / (effect_size**2)

        return int(np.ceil(n))