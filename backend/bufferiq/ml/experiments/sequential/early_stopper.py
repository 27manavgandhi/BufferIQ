"""
Early stopping rules.

Implements various early stopping criteria for experiments.

Example:
```python
    stopper = EarlyStopper()
    
    should_stop = stopper.check(
        control_data=control,
        treatment_data=treatment,
        alpha=0.05,
        min_samples=100
    )
```
"""

from typing import Dict, Optional

import numpy as np

from bufferiq.ml.experiments.statistics.hypothesis_tester import StatisticalAnalyzer
from bufferiq.ml.experiments.design.designer import MetricType


class EarlyStopper:
    """
    Early stopping decision maker.

    Example:
```python
        stopper = EarlyStopper(
            min_samples_per_variant=100,
            futility_threshold=0.01
        )

        result = stopper.check_stopping_criteria(
            control_data=control,
            treatment_data=treatment,
            alpha=0.05
        )

        if result['should_stop']:
            print(f"Reason: {result['reason']}")
```
    """

    def __init__(
        self,
        min_samples_per_variant: int = 100,
        futility_threshold: float = 0.01,
        confidence_threshold: float = 0.99,
    ) -> None:
        """
        Initialize early stopper.

        Args:
            min_samples_per_variant: Minimum samples before stopping
            futility_threshold: Threshold for futility stopping
            confidence_threshold: Confidence for early win
        """
        self.min_samples = min_samples_per_variant
        self.futility_threshold = futility_threshold
        self.confidence_threshold = confidence_threshold
        self.analyzer = StatisticalAnalyzer()

    def check_stopping_criteria(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, any]:
        """
        Check if experiment should stop early.

        Args:
            control_data: Control data
            treatment_data: Treatment data
            alpha: Significance level

        Returns:
            Dictionary with decision and reason
        """
        n_control = len(control_data)
        n_treatment = len(treatment_data)

        # Minimum sample size check
        if n_control < self.min_samples or n_treatment < self.min_samples:
            return {
                "should_stop": False,
                "reason": "Insufficient samples",
                "min_required": self.min_samples,
            }

        # Run hypothesis test
        result = self.analyzer.analyze(
            control_data=control_data,
            treatment_data=treatment_data,
            metric_type=MetricType.ENGAGEMENT_RATE,
            alpha=alpha,
        )

        # Check for clear winner
        if result.is_significant and result.p_value < alpha / 10:  # Very significant
            return {
                "should_stop": True,
                "reason": "clear_winner",
                "winner": "treatment" if result.treatment_mean > result.control_mean else "control",
                "p_value": result.p_value,
            }

        # Check for futility
        if abs(result.relative_diff) < self.futility_threshold:
            return {
                "should_stop": True,
                "reason": "futility",
                "relative_diff": result.relative_diff,
            }

        # Continue experiment
        return {
            "should_stop": False,
            "reason": "continue",
            "p_value": result.p_value,
        }

    def check_futility(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        target_mde: float,
    ) -> bool:
        """
        Check if experiment is futile.

        Args:
            control_data: Control data
            treatment_data: Treatment data
            target_mde: Target minimum detectable effect

        Returns:
            True if futile
        """
        mean_c = float(np.mean(control_data))
        mean_t = float(np.mean(treatment_data))

        if mean_c == 0:
            return False

        actual_diff = abs((mean_t - mean_c) / mean_c)

        return actual_diff < target_mde * 0.5  # Less than half target