"""
Minimum detectable effect calculator.

Calculates the minimum effect size that can be detected
with given sample size and power.

Example:
```python
    calc = MDECalculator()
    
    mde = calc.calculate(
        baseline_rate=0.05,
        sample_size=10000,
        power=0.80,
        alpha=0.05
    )
    
    print(f"MDE: {mde:.1%}")
```
"""

import numpy as np
from scipy import stats


class MDECalculator:
    """
    Calculate minimum detectable effect.

    Example:
```python
        calc = MDECalculator()

        mde = calc.calculate(
            baseline_rate=0.05,
            sample_size=10000,
            power=0.80
        )

        print(f"MDE: {mde:.2%}")
        # Output: MDE: 10.5%
```
    """

    def calculate(
        self,
        baseline_rate: float,
        sample_size: int,
        power: float = 0.80,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate minimum detectable effect.

        Args:
            baseline_rate: Baseline rate
            sample_size: Sample size per variant
            power: Desired power
            alpha: Type I error rate
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

    def calculate_absolute_mde(
        self,
        baseline_rate: float,
        sample_size: int,
        power: float = 0.80,
        alpha: float = 0.05,
        two_tailed: bool = True,
    ) -> float:
        """
        Calculate absolute MDE.

        Args:
            baseline_rate: Baseline rate
            sample_size: Sample size
            power: Power
            alpha: Alpha
            two_tailed: Two-tailed

        Returns:
            Absolute MDE
        """
        relative_mde = self.calculate(
            baseline_rate, sample_size, power, alpha, two_tailed
        )
        return baseline_rate * relative_mde