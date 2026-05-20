"""
Novelty effect detector.

Detects when treatment effects are due to novelty rather than
genuine improvement.

Example:
```python
    detector = NoveltyDetector()
    
    result = detector.detect_novelty(
        daily_metrics=daily_data,
        window_size=7
    )
    
    if result['has_novelty_effect']:
        print(f"Novelty detected: {result['decay_rate']:.2%}")
```
"""

from typing import Dict, List

import numpy as np
from scipy import stats


class NoveltyDetector:
    """
    Detect novelty effects in experiments.

    Example:
```python
        detector = NoveltyDetector(
            significance_level=0.05
        )

        result = detector.detect_novelty(
            daily_treatment_means=[0.10, 0.09, 0.08, 0.07, 0.06],
            daily_control_means=[0.05, 0.05, 0.05, 0.05, 0.05]
        )

        if result['has_novelty_effect']:
            print(f"Novelty effect detected!")
            print(f"Decay rate: {result['decay_rate']:.2%} per day")
```
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        """
        Initialize novelty detector.

        Args:
            significance_level: Significance level for tests
        """
        self.alpha = significance_level

    def detect_novelty(
        self,
        daily_treatment_means: List[float],
        daily_control_means: List[float],
    ) -> Dict[str, any]:
        """
        Detect novelty effect.

        Args:
            daily_treatment_means: Daily means for treatment
            daily_control_means: Daily means for control

        Returns:
            Detection result
        """
        # Calculate daily lifts
        lifts = []
        for t_mean, c_mean in zip(daily_treatment_means, daily_control_means):
            if c_mean > 0:
                lift = (t_mean - c_mean) / c_mean
                lifts.append(lift)

        if len(lifts) < 3:
            return {
                "has_novelty_effect": False,
                "reason": "insufficient_data",
                "days": len(lifts),
            }

        # Test for negative trend (decay)
        days = np.arange(len(lifts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, lifts)

        # Significant negative slope indicates novelty effect
        has_novelty = slope < 0 and p_value < self.alpha

        return {
            "has_novelty_effect": has_novelty,
            "decay_rate": float(slope),
            "initial_lift": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "days_analyzed": len(lifts),
        }

    def estimate_stabilization_time(
        self, daily_treatment_means: List[float], daily_control_means: List[float]
    ) -> int:
        """
        Estimate when effect stabilizes.

        Args:
            daily_treatment_means: Treatment means
            daily_control_means: Control means

        Returns:
            Estimated days to stabilization
        """
        # Calculate daily lifts
        lifts = []
        for t_mean, c_mean in zip(daily_treatment_means, daily_control_means):
            if c_mean > 0:
                lift = (t_mean - c_mean) / c_mean
                lifts.append(lift)

        if len(lifts) < 3:
            return -1

        # Calculate moving average
        window = 3
        moving_avg = []
        for i in range(len(lifts) - window + 1):
            avg = np.mean(lifts[i : i + window])
            moving_avg.append(avg)

        # Find when variance stabilizes
        if len(moving_avg) < 2:
            return -1

        variances = []
        for i in range(1, len(moving_avg)):
            var = np.var(moving_avg[max(0, i - 3) : i + 1])
            variances.append(var)

        # Stabilization when variance is consistently low
        stable_threshold = np.mean(variances) * 0.5
        stable_count = 0
        stable_days = -1

        for i, var in enumerate(variances):
            if var < stable_threshold:
                stable_count += 1
                if stable_count >= 3 and stable_days == -1:
                    stable_days = i + window
            else:
                stable_count = 0

        return stable_days if stable_days > 0 else len(lifts)