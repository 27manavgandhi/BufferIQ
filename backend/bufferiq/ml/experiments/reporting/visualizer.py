"""
Report visualizer.

Creates visualizations for experiment reports.

Example:
```python
    viz = ReportVisualizer()
    
    chart_data = viz.create_comparison_chart(
        control_mean=0.05,
        treatment_mean=0.06
    )
```
"""

from typing import Dict, List

import numpy as np


class ReportVisualizer:
    """
    Create report visualizations.

    Example:
```python
        viz = ReportVisualizer()

        chart = viz.create_comparison_chart(
            control_mean=0.05,
            treatment_mean=0.06,
            ci_lower=0.005,
            ci_upper=0.015
        )

        print(chart)  # Chart data for frontend
```
    """

    def create_comparison_chart(
        self,
        control_mean: float,
        treatment_mean: float,
        ci_lower: float,
        ci_upper: float,
    ) -> Dict:
        """
        Create variant comparison chart data.

        Args:
            control_mean: Control mean
            treatment_mean: Treatment mean
            ci_lower: CI lower bound
            ci_upper: CI upper bound

        Returns:
            Chart data dictionary
        """
        return {
            "type": "bar",
            "data": {
                "labels": ["Control", "Treatment"],
                "datasets": [
                    {
                        "label": "Mean",
                        "data": [control_mean, treatment_mean],
                    }
                ],
            },
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
            },
        }

    def create_time_series_chart(
        self,
        dates: List[str],
        control_values: List[float],
        treatment_values: List[float],
    ) -> Dict:
        """
        Create time series chart data.

        Args:
            dates: Date labels
            control_values: Control time series
            treatment_values: Treatment time series

        Returns:
            Chart data dictionary
        """
        return {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "Control",
                        "data": control_values,
                        "borderColor": "blue",
                    },
                    {
                        "label": "Treatment",
                        "data": treatment_values,
                        "borderColor": "green",
                    },
                ],
            },
        }

    def create_funnel_chart(
        self, steps: List[str], conversion_rates: List[float]
    ) -> Dict:
        """
        Create funnel chart data.

        Args:
            steps: Funnel step names
            conversion_rates: Conversion rates per step

        Returns:
            Chart data dictionary
        """
        return {
            "type": "funnel",
            "data": {
                "labels": steps,
                "datasets": [
                    {
                        "label": "Conversion Rate",
                        "data": conversion_rates,
                    }
                ],
            },
        }