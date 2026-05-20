"""
Power visualizer.

Creates power curves and visualizations.

Example:
```python
    viz = PowerVisualizer()
    
    curve = viz.create_power_curve(
        baseline_rate=0.05,
        sample_sizes=[1000, 5000, 10000, 20000],
        mde=0.10
    )
```
"""

from typing import Dict, List

import numpy as np

from bufferiq.ml.experiments.power.calculator import PowerAnalyzer


class PowerVisualizer:
    """
    Create power visualizations.

    Example:
```python
        viz = PowerVisualizer()

        curve = viz.create_power_curve(
            baseline_rate=0.05,
            sample_sizes=list(range(1000, 20000, 1000)),
            mde=0.10
        )

        print(f"Sample sizes: {curve['sample_sizes']}")
        print(f"Power values: {curve['power_values']}")
```
    """

    def __init__(self) -> None:
        """Initialize visualizer."""
        self.power_analyzer = PowerAnalyzer()

    def create_power_curve(
        self,
        baseline_rate: float,
        sample_sizes: List[int],
        mde: float,
        alpha: float = 0.05,
    ) -> Dict[str, List[float]]:
        """
        Create power curve data.

        Args:
            baseline_rate: Baseline rate
            sample_sizes: List of sample sizes
            mde: Minimum detectable effect
            alpha: Type I error rate

        Returns:
            Dictionary with sample_sizes and power_values
        """
        power_values = []

        for n in sample_sizes:
            power = self.power_analyzer.calculate_power_for_mde(
                baseline_rate=baseline_rate, mde=mde, sample_size=n, alpha=alpha
            )
            power_values.append(power)

        return {"sample_sizes": sample_sizes, "power_values": power_values}

    def create_mde_curve(
        self,
        baseline_rate: float,
        sample_size: int,
        mde_range: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, List[float]]:
        """
        Create MDE curve data.

        Args:
            baseline_rate: Baseline rate
            sample_size: Sample size
            mde_range: List of MDE values
            alpha: Type I error rate

        Returns:
            Dictionary with mde_values and power_values
        """
        power_values = []

        for mde in mde_range:
            power = self.power_analyzer.calculate_power_for_mde(
                baseline_rate=baseline_rate, mde=mde, sample_size=sample_size, alpha=alpha
            )
            power_values.append(power)

        return {"mde_values": mde_range, "power_values": power_values}