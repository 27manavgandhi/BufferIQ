"""
Thompson Sampling bandit algorithm.

Implements Thompson Sampling for multi-armed bandits using
Bayesian probability matching.

Key features:
    - Bayesian approach
    - Beta-Bernoulli model
    - Optimal exploration-exploitation
    - No hyperparameters to tune

Example:
```python
    ts = ThompsonSampling()
    
    arms = [
        BanditArm("control", "Original"),
        BanditArm("treatment_a", "Variant A"),
        BanditArm("treatment_b", "Variant B")
    ]
    
    # Select arm
    selected = ts.select_arm(arms)
    
    # Update with reward
    ts.update(selected, reward=1)
```
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BanditArm:
    """Multi-armed bandit arm (variant)."""

    variant_id: str
    variant_name: str

    # Performance
    successes: int = 0
    trials: int = 0
    mean_reward: float = 0.0

    # Thompson Sampling parameters (Beta distribution)
    alpha: float = 1.0  # Prior successes + observed successes
    beta: float = 1.0  # Prior failures + observed failures

    def __post_init__(self) -> None:
        """Initialize derived fields."""
        if self.trials > 0:
            self.mean_reward = self.successes / self.trials


class ThompsonSampling:
    """
    Thompson Sampling bandit algorithm.

    Balances exploration and exploitation using
    Bayesian probability matching.

    Example:
```python
        ts = ThompsonSampling()

        # Initialize arms
        arms = [
            BanditArm("control", "Original"),
            BanditArm("treatment_a", "Variant A"),
            BanditArm("treatment_b", "Variant B")
        ]

        # Select arm
        selected = ts.select_arm(arms)
        print(f"Selected: {selected.variant_name}")

        # Update with reward
        ts.update(selected, reward=1)  # Success

        # Continue selecting and updating...
        for _ in range(100):
            arm = ts.select_arm(arms)
            reward = simulate_reward(arm)  # Your reward function
            ts.update(arm, reward=reward)

        # Check final performance
        for arm in arms:
            print(f"{arm.variant_name}: {arm.mean_reward:.3f}")
```
    """

    def select_arm(self, arms: List[BanditArm]) -> BanditArm:
        """
        Select arm using Thompson Sampling.

        Samples from each arm's posterior distribution and
        selects the arm with the highest sample.

        Args:
            arms: List of bandit arms

        Returns:
            Selected arm
        """
        if not arms:
            raise ValueError("No arms provided")

        # Sample from each arm's posterior (Beta distribution)
        samples = []
        for arm in arms:
            sample = np.random.beta(arm.alpha, arm.beta)
            samples.append(sample)

        # Select arm with highest sample
        best_idx = np.argmax(samples)
        return arms[best_idx]

    def update(self, arm: BanditArm, reward: float) -> None:
        """
        Update arm parameters with observed reward.

        Args:
            arm: Arm to update
            reward: Observed reward (0 or 1 for binary)
        """
        arm.trials += 1

        if reward > 0:
            arm.successes += 1
            arm.alpha += 1
        else:
            arm.beta += 1

        # Update mean reward
        arm.mean_reward = arm.successes / arm.trials if arm.trials > 0 else 0.0

    def get_arm_statistics(self, arm: BanditArm) -> dict:
        """
        Get statistics for an arm.

        Args:
            arm: Bandit arm

        Returns:
            Dictionary with statistics
        """
        # Expected value (mean of Beta distribution)
        expected_value = arm.alpha / (arm.alpha + arm.beta)

        # Variance
        variance = (arm.alpha * arm.beta) / (
            (arm.alpha + arm.beta) ** 2 * (arm.alpha + arm.beta + 1)
        )

        # 95% credible interval
        ci_lower = float(np.percentile(np.random.beta(arm.alpha, arm.beta, 10000), 2.5))
        ci_upper = float(np.percentile(np.random.beta(arm.alpha, arm.beta, 10000), 97.5))

        return {
            "expected_value": float(expected_value),
            "variance": float(variance),
            "trials": arm.trials,
            "successes": arm.successes,
            "mean_reward": arm.mean_reward,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def calculate_probability_best(
        self, arms: List[BanditArm], n_samples: int = 10000
    ) -> dict:
        """
        Calculate probability each arm is best.

        Args:
            arms: List of arms
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary mapping arm variant_id to probability
        """
        # Sample from posteriors
        samples = []
        for arm in arms:
            arm_samples = np.random.beta(arm.alpha, arm.beta, n_samples)
            samples.append(arm_samples)

        samples = np.array(samples)  # Shape: (n_arms, n_samples)

        # Count how often each arm is best
        best_counts = np.sum(samples == samples.max(axis=0), axis=1)

        # Calculate probabilities
        probabilities = {}
        for arm, count in zip(arms, best_counts):
            probabilities[arm.variant_id] = float(count / n_samples)

        return probabilities

    def calculate_regret(self, arms: List[BanditArm], true_best_mean: float) -> float:
        """
        Calculate cumulative regret.

        Args:
            arms: List of arms
            true_best_mean: True mean of best arm

        Returns:
            Cumulative regret
        """
        total_trials = sum(arm.trials for arm in arms)
        total_reward = sum(arm.successes for arm in arms)

        optimal_reward = true_best_mean * total_trials
        actual_reward = total_reward

        regret = optimal_reward - actual_reward

        return float(max(0, regret))