"""
A/B tester for hashtags.

Runs statistical tests to compare hashtag performance.
"""

from typing import Dict, List, Tuple
from scipy import stats
import numpy as np


class ABTester:
    """
    Run A/B tests for hashtags.

    Uses t-tests to determine statistical significance.

    Example:
```python
        tester = ABTester()
        result = tester.run_test(
            with_hashtag=[150, 160, 140, 155, 165],
            without_hashtag=[120, 125, 115, 130, 118]
        )

        if result['is_significant']:
            print(f"Significant difference! p={result['p_value']:.4f}")
            print(f"Effect size: {result['effect_size']:.2f}")
```
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Initialize A/B tester.

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha

    def run_test(
        self, with_hashtag: List[float], without_hashtag: List[float]
    ) -> Dict[str, float | bool | str]:
        """
        Run independent t-test.

        Args:
            with_hashtag: Engagement values with hashtag
            without_hashtag: Engagement values without hashtag

        Returns:
            Test results including significance and effect size
        """
        if len(with_hashtag) < 2 or len(without_hashtag) < 2:
            return {
                "is_significant": False,
                "p_value": 1.0,
                "t_statistic": 0.0,
                "effect_size": 0.0,
                "recommendation": "insufficient_data",
            }

        # Run t-test
        t_stat, p_value = stats.ttest_ind(with_hashtag, without_hashtag)

        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(with_hashtag, without_hashtag)

        # Determine significance
        is_significant = p_value < self.alpha

        # Recommendation
        recommendation = self._get_recommendation(is_significant, effect_size)

        return {
            "is_significant": is_significant,
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "effect_size": float(effect_size),
            "recommendation": recommendation,
            "confidence_level": float(1 - p_value),
        }

    def _calculate_cohens_d(
        self, group1: List[float], group2: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            Cohen's d effect size
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        return cohens_d

    def _get_recommendation(self, is_significant: bool, effect_size: float) -> str:
        """
        Get recommendation based on test results.

        Args:
            is_significant: Whether result is statistically significant
            effect_size: Cohen's d effect size

        Returns:
            Recommendation string
        """
        if not is_significant:
            return "no_significant_difference"

        # Effect size interpretation:
        # Small: 0.2, Medium: 0.5, Large: 0.8
        abs_effect = abs(effect_size)

        if abs_effect >= 0.8:
            return "strong_effect" if effect_size > 0 else "strong_negative_effect"
        elif abs_effect >= 0.5:
            return "moderate_effect" if effect_size > 0 else "moderate_negative_effect"
        elif abs_effect >= 0.2:
            return "small_effect" if effect_size > 0 else "small_negative_effect"
        else:
            return "negligible_effect"