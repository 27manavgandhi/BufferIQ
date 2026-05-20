"""
Upper Confidence Bound (UCB) bandit algorithm.

Implements UCB1 algorithm for multi-armed bandits.

Example:
```python
    ucb = UCB(exploration_param=2.0)
    
    arm = ucb.select_arm(arms, total_trials=100)
    ucb.update(arm, reward=1)
```
"""

from typing import List

import numpy as np

from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class UCB:
    """
    Upper Confidence Bound algorithm.

    Selects arm with highest upper confidence bound on mean reward.

    Example:
```python
        ucb = UCB(exploration_param=2.0)

        arms = [
            BanditArm("control", "Original"),
            BanditArm("treatment", "New")
        ]

        total_trials = 0
        for _ in range(1000):
            arm = ucb.select_arm(arms, total_trials)
            reward = simulate_reward(arm)
            ucb.update(arm, reward)
            total_trials += 1

        # Best arm
        best = max(arms, key=lambda a: a.mean_reward)
        print(f"Best: {best.variant_name}")
```
    """

    def __init__(self, exploration_param: float = 2.0) -> None:
        """
        Initialize UCB.

        Args:
            exploration_param: Exploration parameter (typically 2.0)
        """
        self.c = exploration_param

    def select_arm(self, arms: List[BanditArm], total_trials: int) -> BanditArm:
        """
        Select arm with highest UCB.

        Args:
            arms: List of arms
            total_trials: Total trials across all arms

        Returns:
            Selected arm
        """
        if not arms:
            raise ValueError("No arms provided")

        # Select arms with zero trials first (exploration)
        untried = [arm for arm in arms if arm.trials == 0]
        if untried:
            return untried[0]

        # Calculate UCB for each arm
        ucb_values = []
        for arm in arms:
            mean = arm.mean_reward

            # Exploration bonus
            if arm.trials > 0 and total_trials > 0:
                exploration = np.sqrt((self.c * np.log(total_trials)) / arm.trials)
            else:
                exploration = 0.0

            ucb = mean + exploration
            ucb_values.append(ucb)

        # Select arm with highest UCB
        best_idx = np.argmax(ucb_values)
        return arms[best_idx]

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

    def get_ucb_value(self, arm: BanditArm, total_trials: int) -> float:
        """
        Calculate UCB value for arm.

        Args:
            arm: Bandit arm
            total_trials: Total trials

        Returns:
            UCB value
        """
        if arm.trials == 0:
            return float("inf")

        mean = arm.mean_reward
        exploration = np.sqrt((self.c * np.log(total_trials)) / arm.trials)

        return float(mean + exploration)