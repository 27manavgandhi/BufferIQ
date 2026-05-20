"""
Sample size calculator.

Calculates required sample size for experiments using
power analysis and statistical formulas.

Key features:
    - Power analysis
    - Sample size calculation
    - Effect size computation
    - Bonferroni correction
    - Multiple comparison adjustment

Example:
```python
    calculator = SampleSizeCalculator()
    
    sample_size = calculator.calculate(
        baseline_rate=0.05,  # 5% engagement rate
        mde=0.10,  # 10% relative change
        alpha=0.05,
        power=0.80
    )
    
    print(f"Required sample size: {sample_size:,} per variant")
    # Output: Required sample size: 15,684 per variant
```
"""

from typing import Optional

import numpy as np
from scipy import stats


class SampleSizeCalculator:
    """
    Calculate required sample size for experiments.

    Uses power analysis to determine sample size needed
    to detect minimum effect with desired power.

    Example:
```python
        calculator = SampleSizeCalculator()

        sample_size = calculator.calculate(
            baseline_rate=0.05,
            mde=0.10,
            alpha=0.05,
            power=0.80
        )

        print(f"Required: {sample_size:,} per variant")
```
    """

    def calculate(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
        num_variants: int = 2,
        two_tailed: bool = True,
    ) -> int:
        """
        Calculate required sample size per variant.

        Uses Cohen's h for proportions and applies Bonferroni
        correction for multiple comparisons.

        Args:
            baseline_rate: Baseline conversion rate (0-1)
            mde: Minimum detectable effect (relative change)
            alpha: Type I error rate
            power: Statistical power (1 - Type II error)
            num_variants: Number of variants
            two_tailed: Use two-tailed test

        Returns:
            Required sample size per variant

        Raises:
            ValueError: If parameters invalid
        """
        # Validate inputs
        if not 0 < baseline_rate < 1:
            raise ValueError(
                f"Baseline rate must be between 0 and 1, got {baseline_rate}"
            )
        if not 0 < mde < 1:
            raise ValueError(f"MDE must be between 0 and 1, got {mde}")
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"Power must be between 0 and 1, got {power}")
        if num_variants < 2:
            raise ValueError(f"At least 2 variants required, got {num_variants}")

        # Calculate treatment rate
        treatment_rate = baseline_rate * (1 + mde)

        # Ensure treatment rate is valid
        if treatment_rate >= 1:
            treatment_rate = 0.99

        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = treatment_rate
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Handle zero effect size
        if abs(effect_size) < 1e-10:
            raise ValueError("Effect size too small to detect")

        # Z-scores
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Sample size per variant
        n = ((z_alpha + z_beta) ** 2) / (effect_size**2)

        # Bonferroni correction for multiple comparisons
        if num_variants > 2:
            # Adjust alpha for multiple comparisons
            alpha_adjusted = alpha / (num_variants - 1)

            if two_tailed:
                z_alpha_adjusted = stats.norm.ppf(1 - alpha_adjusted / 2)
            else:
                z_alpha_adjusted = stats.norm.ppf(1 - alpha_adjusted)

            n = ((z_alpha_adjusted + z_beta) ** 2) / (effect_size**2)

        return int(np.ceil(n))

    def calculate_power(
        self,
        baseline_rate: float,
        treatment_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate statistical power for given sample size.

        Args:
            baseline_rate: Baseline rate
            treatment_rate: Treatment rate
            sample_size: Sample size per variant
            alpha: Type I error rate
            two_tailed: Two-tailed test

        Returns:
            Statistical power (0-1)
        """
        # Effect size
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

        return float(power)

    def calculate_mde(
        self,
        baseline_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.80,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate minimum detectable effect for given sample size.

        Args:
            baseline_rate: Baseline rate
            sample_size: Sample size per variant
            alpha: Type I error rate
            power: Desired power
            two_tailed: Two-tailed test

        Returns:
            Minimum detectable effect (relative change)
        """
        # Z-scores
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Effect size
        effect_size = (z_alpha + z_beta) / np.sqrt(sample_size)

        # Convert to treatment rate
        h = effect_size
        treatment_rate = (np.sin(np.arcsin(np.sqrt(baseline_rate)) + h / 2)) ** 2

        # Calculate relative MDE
        mde = (treatment_rate - baseline_rate) / baseline_rate

        return float(abs(mde))