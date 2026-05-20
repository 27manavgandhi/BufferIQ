"""
Interaction effect detector.

Detects interactions between treatment and other factors like
user segments, time periods, or platforms.

Key features:
    - Segment-level interaction detection
    - Time-based interaction effects
    - Platform interaction analysis
    - Statistical significance testing

Example:
```python
    detector = InteractionDetector()
    
    result = detector.detect_interaction(
        segment_treatment_means={"new": 0.12, "old": 0.08},
        segment_control_means={"new": 0.05, "old": 0.05},
        segment_sizes={"new": 1000, "old": 1000}
    )
    
    if result['has_interaction']:
        print("Treatment effect varies by segment!")
```
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


class InteractionDetector:
    """
    Detect interaction effects in experiments.

    Checks if treatment effects vary across subgroups or factors.

    Example:
```python
        detector = InteractionDetector()

        # Check if treatment effect varies by segment
        result = detector.detect_interaction(
            segment_treatment_means={"new": 0.12, "old": 0.08},
            segment_control_means={"new": 0.05, "old": 0.05},
            segment_sizes={"new": 1000, "old": 1000}
        )

        if result['has_interaction']:
            print("Treatment effect varies by segment!")
            print(f"Segment lifts: {result['segment_lifts']}")
```
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Initialize interaction detector.

        Args:
            alpha: Significance level for tests
        """
        self.alpha = alpha

    def detect_interaction(
        self,
        segment_treatment_means: Dict[str, float],
        segment_control_means: Dict[str, float],
        segment_sizes: Dict[str, int],
        segment_variances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Detect interaction effect between treatment and segment.

        Args:
            segment_treatment_means: Treatment means by segment
            segment_control_means: Control means by segment
            segment_sizes: Sample sizes by segment
            segment_variances: Optional variances by segment

        Returns:
            Interaction detection result
        """
        # Validate inputs
        if set(segment_treatment_means.keys()) != set(segment_control_means.keys()):
            raise ValueError("Segment keys must match")

        if len(segment_treatment_means) < 2:
            return {
                "has_interaction": False,
                "reason": "insufficient_segments",
                "min_required": 2,
            }

        # Calculate lifts by segment
        segment_lifts = {}
        for segment in segment_treatment_means.keys():
            t_mean = segment_treatment_means[segment]
            c_mean = segment_control_means[segment]

            if c_mean > 0:
                lift = (t_mean - c_mean) / c_mean
            else:
                lift = 0.0

            segment_lifts[segment] = lift

        # Test for significant difference in lifts
        lift_values = list(segment_lifts.values())

        # Variance in lifts
        lift_variance = float(np.var(lift_values))
        lift_std = float(np.std(lift_values))

        # F-test for variance (simplified)
        # If variance is significantly large, interaction exists
        mean_lift = float(np.mean(lift_values))

        # Calculate heterogeneity statistic
        heterogeneity = lift_std / abs(mean_lift) if abs(mean_lift) > 1e-6 else 0.0

        # Interaction detected if:
        # 1. High variance in lifts across segments
        # 2. Lifts differ by more than threshold
        threshold = 0.05  # 5% absolute difference

        max_lift = max(lift_values)
        min_lift = min(lift_values)
        lift_range = max_lift - min_lift

        has_interaction = lift_range > threshold and lift_variance > 0.001

        # If variances provided, do formal test
        p_value = None
        if segment_variances and len(segment_treatment_means) == 2:
            # Two-sample test for difference in treatment effects
            segments = list(segment_treatment_means.keys())
            seg1, seg2 = segments[0], segments[1]

            # Treatment effects
            effect1 = segment_treatment_means[seg1] - segment_control_means[seg1]
            effect2 = segment_treatment_means[seg2] - segment_control_means[seg2]

            # Standard errors (simplified)
            var1 = segment_variances.get(seg1, lift_variance)
            var2 = segment_variances.get(seg2, lift_variance)
            n1 = segment_sizes[seg1]
            n2 = segment_sizes[seg2]

            se1 = np.sqrt(var1 / n1)
            se2 = np.sqrt(var2 / n2)
            se_diff = np.sqrt(se1**2 + se2**2)

            if se_diff > 0:
                z_stat = (effect1 - effect2) / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                has_interaction = p_value < self.alpha

        return {
            "has_interaction": has_interaction,
            "segment_lifts": segment_lifts,
            "lift_variance": lift_variance,
            "lift_std": lift_std,
            "mean_lift": mean_lift,
            "min_lift": float(min_lift),
            "max_lift": float(max_lift),
            "lift_range": float(lift_range),
            "heterogeneity": float(heterogeneity),
            "p_value": float(p_value) if p_value is not None else None,
            "recommendation": self._generate_recommendation(
                has_interaction, segment_lifts
            ),
        }

    def _generate_recommendation(
        self, has_interaction: bool, segment_lifts: Dict[str, float]
    ) -> str:
        """
        Generate recommendation based on interaction.

        Args:
            has_interaction: Whether interaction detected
            segment_lifts: Lifts by segment

        Returns:
            Recommendation string
        """
        if not has_interaction:
            return "Treatment effect is consistent across segments"

        # Find best and worst segments
        best_segment = max(segment_lifts.items(), key=lambda x: x[1])
        worst_segment = min(segment_lifts.items(), key=lambda x: x[1])

        return (
            f"Treatment effect varies by segment. "
            f"Best: {best_segment[0]} ({best_segment[1]:.2%}), "
            f"Worst: {worst_segment[0]} ({worst_segment[1]:.2%}). "
            f"Consider segment-specific strategies."
        )

    def detect_time_interaction(
        self,
        daily_treatment_effects: List[float],
        window_size: int = 7,
    ) -> Dict[str, any]:
        """
        Detect time-based interaction (treatment effect changes over time).

        Args:
            daily_treatment_effects: Daily treatment effects
            window_size: Window for moving average

        Returns:
            Time interaction result
        """
        if len(daily_treatment_effects) < window_size:
            return {
                "has_time_interaction": False,
                "reason": "insufficient_days",
            }

        # Calculate moving average
        moving_avg = []
        for i in range(len(daily_treatment_effects) - window_size + 1):
            avg = np.mean(daily_treatment_effects[i : i + window_size])
            moving_avg.append(avg)

        # Test for trend
        if len(moving_avg) < 3:
            return {
                "has_time_interaction": False,
                "reason": "insufficient_windows",
            }

        days = np.arange(len(moving_avg))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            days, moving_avg
        )

        has_time_interaction = p_value < self.alpha and abs(slope) > 0.001

        return {
            "has_time_interaction": has_time_interaction,
            "trend_slope": float(slope),
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "initial_effect": float(moving_avg[0]),
            "final_effect": float(moving_avg[-1]),
            "effect_change": float(moving_avg[-1] - moving_avg[0]),
        }

    def detect_platform_interaction(
        self,
        platform_treatment_means: Dict[str, float],
        platform_control_means: Dict[str, float],
        platform_sizes: Dict[str, int],
    ) -> Dict[str, any]:
        """
        Detect platform-specific interaction.

        Args:
            platform_treatment_means: Treatment means by platform
            platform_control_means: Control means by platform
            platform_sizes: Sample sizes by platform

        Returns:
            Platform interaction result
        """
        # Reuse segment interaction detection
        return self.detect_interaction(
            segment_treatment_means=platform_treatment_means,
            segment_control_means=platform_control_means,
            segment_sizes=platform_sizes,
        )