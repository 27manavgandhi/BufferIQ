"""
Sample Ratio Mismatch (SRM) detector.

Detects when traffic allocation doesn't match expected ratios.

Example:
```python
    detector = SRMDetector()
    
    result = detector.detect_srm(
        variant_counts={"control": 1000, "treatment": 900},
        expected_ratios={"control": 0.5, "treatment": 0.5}
    )
```
"""

from typing import Dict

import numpy as np
from scipy import stats


class SRMDetector:
    """
    Detect sample ratio mismatch.

    Example:
```python
        detector = SRMDetector(alpha=0.001)

        result = detector.detect_srm(
            variant_counts={"control": 1020, "treatment": 980},
            expected_ratios={"control": 0.5, "treatment": 0.5}
        )

        if result['has_srm']:
            print("SRM detected! Traffic allocation is off.")
```
    """

    def __init__(self, alpha: float = 0.001) -> None:
        """
        Initialize SRM detector.

        Args:
            alpha: Significance level (typically very low)
        """
        self.alpha = alpha

    def detect_srm(
        self,
        variant_counts: Dict[str, int],
        expected_ratios: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Detect sample ratio mismatch using chi-square test.

        Args:
            variant_counts: Actual counts per variant
            expected_ratios: Expected traffic ratios

        Returns:
            SRM detection result
        """
        # Validate inputs
        if set(variant_counts.keys()) != set(expected_ratios.keys()):
            raise ValueError("Variant keys must match")

        # Calculate expected counts
        total = sum(variant_counts.values())
        expected_counts = {
            variant: total * ratio for variant, ratio in expected_ratios.items()
        }

        # Prepare for chi-square test
        observed = np.array([variant_counts[v] for v in sorted(variant_counts.keys())])
        expected = np.array([expected_counts[v] for v in sorted(expected_counts.keys())])

        # Chi-square test
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(variant_counts) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        # SRM detected if p-value is very small
        has_srm = p_value < self.alpha

        # Calculate deviations
        deviations = {}
        for variant in variant_counts.keys():
            actual = variant_counts[variant]
            expected_val = expected_counts[variant]
            deviation = (actual - expected_val) / expected_val if expected_val > 0 else 0
            deviations[variant] = deviation

        return {
            "has_srm": has_srm,
            "chi2_statistic": float(chi2_stat),
            "chi2_p_value": float(p_value),
            "degrees_of_freedom": df,
            "observed_counts": variant_counts,
            "expected_counts": {k: float(v) for k, v in expected_counts.items()},
            "deviations": {k: float(v) for k, v in deviations.items()},
        }

    def calculate_required_sample_size_for_srm(
        self,
        expected_ratios: Dict[str, float],
        min_detectable_deviation: float = 0.01,
    ) -> int:
        """
        Calculate sample size needed to detect SRM.

        Args:
            expected_ratios: Expected traffic ratios
            min_detectable_deviation: Minimum deviation to detect

        Returns:
            Required total sample size
        """
        # Simplified calculation
        # Larger samples detect smaller deviations
        num_variants = len(expected_ratios)

        # Rough estimate based on chi-square power
        base_n = 1000 * num_variants
        adjustment = 1 / (min_detectable_deviation ** 2)

        required_n = int(base_n * adjustment)

        return required_n