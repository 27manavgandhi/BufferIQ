"""
Power analysis module.

Calculates statistical power, minimum detectable effect,
and generates power curves.

Components:
    - PowerAnalyzer: Power calculation
    - MDECalculator: Minimum detectable effect
    - PowerVisualizer: Power curves

Example:
```python
    from bufferiq.ml.experiments.power import PowerAnalyzer
    
    analyzer = PowerAnalyzer()
    
    power = analyzer.calculate_power(
        baseline_rate=0.05,
        treatment_rate=0.055,
        sample_size=10000
    )
```
"""

from bufferiq.ml.experiments.power.calculator import PowerAnalyzer
from bufferiq.ml.experiments.power.mde_calculator import MDECalculator
from bufferiq.ml.experiments.power.visualizer import PowerVisualizer

__all__ = [
    "PowerAnalyzer",
    "MDECalculator",
    "PowerVisualizer",
]