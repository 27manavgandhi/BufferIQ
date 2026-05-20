"""
Bandit algorithms module.

Implements multi-armed bandit algorithms for adaptive experimentation.

Components:
    - ThompsonSampling: Bayesian bandit
    - UCB: Upper Confidence Bound
    - EpsilonGreedy: Epsilon-greedy exploration
    - ContextualBandit: Contextual bandits

Example:
```python
    from bufferiq.ml.experiments.bandits import ThompsonSampling
    
    ts = ThompsonSampling()
    
    arm = ts.select_arm(arms)
    ts.update(arm, reward=1)
```
"""

from bufferiq.ml.experiments.bandits.thompson_sampling import (
    ThompsonSampling,
    BanditArm,
)
from bufferiq.ml.experiments.bandits.ucb import UCB
from bufferiq.ml.experiments.bandits.epsilon_greedy import EpsilonGreedy
from bufferiq.ml.experiments.bandits.contextual import ContextualBandit

__all__ = [
    "ThompsonSampling",
    "BanditArm",
    "UCB",
    "EpsilonGreedy",
    "ContextualBandit",
]