"""
Statistical analysis module.

Performs hypothesis testing, confidence intervals, effect size
calculation, and Bayesian analysis.

Components:
    - HypothesisTester: Statistical tests
    - ConfidenceInterval: CI calculation
    - EffectSize: Effect size metrics
    - BayesianAnalyzer: Bayesian inference

Example:
```python
    from bufferiq.ml.experiments.statistics import StatisticalAnalyzer
    
    analyzer = StatisticalAnalyzer()
    
    result = analyzer.analyze(
        control_data=control,
        treatment_data=treatment,
        metric_type=MetricType.ENGAGEMENT_RATE
    )
```
"""

from bufferiq.ml.experiments.statistics.hypothesis_tester import (
    StatisticalAnalyzer,
    HypothesisTestResult,
    BayesianResult,
)
from bufferiq.ml.experiments.statistics.confidence_interval import (
    ConfidenceIntervalCalculator,
)
from bufferiq.ml.experiments.statistics.effect_size import EffectSizeCalculator
from bufferiq.ml.experiments.statistics.bayesian_analyzer import BayesianAnalyzer

__all__ = [
    "StatisticalAnalyzer",
    "HypothesisTestResult",
    "BayesianResult",
    "ConfidenceIntervalCalculator",
    "EffectSizeCalculator",
    "BayesianAnalyzer",
]