"""
Time-based analyzer.

Analyzes time-based patterns and trends in experiment data.

Example:
```python
    analyzer = TimeAnalyzer()
    
    trend = analyzer.analyze_trend(
        daily_data=data,
        metric_name="engagement"
    )
```
"""

from typing import Dict, List

import numpy as np
from scipy import stats


class TimeAnalyzer:
    """
    Analyze time-based patterns.

    Example:
```python
        analyzer = TimeAnalyzer()

        trend = analyzer.analyze_trend(
            time_series=[0.10, 0.11, 0.09, 0.12, 0.10],
            timestamps=["2024-01-01", "2024-01-02", ...]
        )

        print(f"Trend: {trend['direction']}")
        print(f"Slope: {trend['slope']:.4f}")
```
    """

    def analyze_trend(
        self, time_series: List[float], alpha: float = 0.05
    ) -> Dict[str, any]:
        """
        Analyze trend in time series.

        Args:
            time_series: Time series data
            alpha: Significance level

        Returns:
            Trend analysis
        """
        if len(time_series) < 2:
            return {"direction": "unknown", "is_significant": False}

        # Linear regression
        x = np.arange(len(time_series))
        y = np.array(time_series)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine direction
        if p_value < alpha:
            if slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
        else:
            direction = "flat"

        return {
            "direction": direction,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "is_significant": p_value < alpha,
        }

    def detect_changepoint(
        self, time_series: List[float], min_segment_length: int = 3
    ) -> Dict[str, any]:
        """
        Detect changepoint in time series.

        Args:
            time_series: Time series data
            min_segment_length: Minimum segment length

        Returns:
            Changepoint detection result
        """
        if len(time_series) < min_segment_length * 2:
            return {"has_changepoint": False}

        # Simple changepoint detection using variance
        best_split = -1
        min_variance = float("inf")

        for i in range(min_segment_length, len(time_series) - min_segment_length):
            left_var = np.var(time_series[:i])
            right_var = np.var(time_series[i:])
            total_var = left_var + right_var

            if total_var < min_variance:
                min_variance = total_var
                best_split = i

        # Check if changepoint is significant
        if best_split > 0:
            left_mean = np.mean(time_series[:best_split])
            right_mean = np.mean(time_series[best_split:])

            # T-test
            t_stat, p_value = stats.ttest_ind(
                time_series[:best_split], time_series[best_split:]
            )

            return {
                "has_changepoint": p_value < 0.05,
                "changepoint_index": best_split,
                "left_mean": float(left_mean),
                "right_mean": float(right_mean),
                "p_value": float(p_value),
            }

        return {"has_changepoint": False}