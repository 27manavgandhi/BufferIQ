"""
Sequential testing module.

Implements sequential probability ratio test (SPRT) and
early stopping rules for experiments.

Components:
    - SequentialTester: SPRT implementation
    - EarlyStopper: Early stopping rules
    - GroupSequential: Group sequential designs

Example:
```python
    from bufferiq.ml.experiments.sequential import SequentialTester
    
    tester = SequentialTester()
    
    result = tester.test(
        control_successes=50,
        control_trials=1000,
        treatment_successes=60,
        treatment_trials=1000
    )
```
"""

from bufferiq.ml.experiments.sequential.sprt import SequentialTester
from bufferiq.ml.experiments.sequential.early_stopper import EarlyStopper
from bufferiq.ml.experiments.sequential.group_sequential import GroupSequentialTester

__all__ = [
    "SequentialTester",
    "EarlyStopper",
    "GroupSequentialTester",
]