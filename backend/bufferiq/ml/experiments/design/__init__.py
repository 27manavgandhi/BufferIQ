"""
Experiment design module.

Handles experiment configuration, sample size calculation,
stratification, and randomization strategies.

Components:
    - ExperimentDesigner: Configure A/B and multivariate tests
    - SampleSizeCalculator: Power analysis and sample size
    - Stratifier: Stratified randomization
    - Randomizer: Assignment strategies

Example:
```python
    from bufferiq.ml.experiments.design import ExperimentDesigner
    
    designer = ExperimentDesigner()
    
    config = designer.design(
        name="Headline Test",
        description="Test AI headlines",
        variants=[control, treatment],
        platform="linkedin",
        primary_metric=MetricType.ENGAGEMENT_RATE,
        baseline_rate=0.05,
        mde=0.10
    )
```
"""

from bufferiq.ml.experiments.design.designer import (
    ExperimentDesigner,
    ExperimentConfig,
    Variant,
    ExperimentType,
    MetricType,
)
from bufferiq.ml.experiments.design.sample_size_calculator import (
    SampleSizeCalculator,
)
from bufferiq.ml.experiments.design.stratifier import Stratifier
from bufferiq.ml.experiments.design.randomizer import Randomizer

__all__ = [
    "ExperimentDesigner",
    "ExperimentConfig",
    "Variant",
    "ExperimentType",
    "MetricType",
    "SampleSizeCalculator",
    "Stratifier",
    "Randomizer",
]