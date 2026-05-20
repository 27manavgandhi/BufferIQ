"""
Epsilon-Greedy bandit algorithm.

Implements epsilon-greedy exploration strategy.

Example:
```python
    eg = EpsilonGreedy(epsilon=0.1)
    
    arm = eg.select_arm(arms)
    eg.update(arm, reward=1)
```
"""

from typing import List

import numpy as np

from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class EpsilonGreedy:
    """
    Epsilon-Greedy algorithm.

    Explores with probability epsilon, exploits otherwise.

    Example:
```python
        eg = EpsilonGreedy(epsilon=0.1)

        arms = [
            BanditArm("control", "Original"),
            BanditArm("treatment", "New")
        ]

        for _ in range(1000):
            arm = eg.select_arm(arms)
            reward = simulate_reward(arm)
            eg.update(arm, reward)

        # Best arm
        best = max(arms, key=lambda a: a.mean_reward)
        print(f"Best: {best.variant_name}")
```
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        """
        Initialize epsilon-greedy.

        Args:
            epsilon: Exploration probability (0-1)
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")

        self.epsilon = epsilon

    def select_arm(self, arms: List[BanditArm]) -> BanditArm:
        """
        Select arm using epsilon-greedy.

        Args:
            arms: List of arms

        Returns:
            Selected arm
        """
        if not arms:
            raise ValueError("No arms provided")

        # Explore
        if np.random.random() < self.epsilon:
            return np.random.choice(arms)

        # Exploit - select arm with highest mean
        untried = [arm for arm in arms if arm.trials == 0]
        if untried:
            return untried[0]

        best_arm = max(arms, key=lambda a: a.mean_reward)
        return best_arm

    def update(self, arm: BanditArm, reward: float) -> None:
        """
        Update arm with reward.

        Args:
            arm: Arm to update
            reward: Observed reward
        """
        arm.trials += 1

        if reward > 0:
            arm.successes += 1

        # Update mean reward
        arm.mean_reward = arm.successes / arm.trials if arm.trials > 0 else 0.0

    def set_epsilon(self, epsilon: float) -> None:
        """
        Update epsilon value.

        Args:
            epsilon: New epsilon value
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")

        self.epsilon = epsilon

    def decay_epsilon(self, decay_rate: float = 0.99) -> None:
        """
        Decay epsilon over time.

        Args:
            decay_rate: Decay multiplier (0-1)
        """
        self.epsilon = max(0.01, self.epsilon * decay_rate)