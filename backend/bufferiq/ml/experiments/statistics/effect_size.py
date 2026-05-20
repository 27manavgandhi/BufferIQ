"""
Effect size calculator.

Calculates various effect size metrics including Cohen's d,
Hedge's g, and Cliff's delta.

Example:
```python
    calc = EffectSizeCalculator()
    
    effect_size = calc.cohens_d(control, treatment)
```
"""

import numpy as np


class EffectSizeCalculator:
    """
    Calculate effect sizes.

    Example:
```python
        calc = EffectSizeCalculator()

        cohen_d = calc.cohens_d(
            control=[100, 105, 98],
            treatment=[110, 115, 108]
        )

        print(f"Cohen's d: {cohen_d:.2f}")
```
    """

    def cohens_d(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            control: Control data
            treatment: Treatment data

        Returns:
            Cohen's d
        """
        # Calculate means
        mean_c = np.mean(control)
        mean_t = np.mean(treatment)

        # Pooled standard deviation
        n_c = len(control)
        n_t = len(treatment)

        var_c = np.var(control, ddof=1)
        var_t = np.var(treatment, ddof=1)

        pooled_std = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0.0

        d = (mean_t - mean_c) / pooled_std

        return float(d)

    def hedges_g(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Calculate Hedge's g (bias-corrected Cohen's d).

        Args:
            control: Control data
            treatment: Treatment data

        Returns:
            Hedge's g
        """
        # Calculate Cohen's d
        d = self.cohens_d(control, treatment)

        # Bias correction factor
        n_c = len(control)
        n_t = len(treatment)
        n = n_c + n_t

        correction = 1 - (3 / (4 * n - 9))

        g = d * correction

        return float(g)

    def cliffs_delta(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Calculate Cliff's delta (non-parametric effect size).

        Args:
            control: Control data
            treatment: Treatment data

        Returns:
            Cliff's delta (-1 to 1)
        """
        n_c = len(control)
        n_t = len(treatment)

        # Count dominances
        dominances = 0
        for t_val in treatment:
            for c_val in control:
                if t_val > c_val:
                    dominances += 1
                elif t_val < c_val:
                    dominances -= 1

        # Cliff's delta
        delta = dominances / (n_c * n_t)

        return float(delta)

    def cohens_h(self, p1: float, p2: float) -> float:
        """
        Calculate Cohen's h for proportions.

        Args:
            p1: Proportion 1
            p2: Proportion 2

        Returns:
            Cohen's h
        """
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        return float(h)