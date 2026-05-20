"""
Contextual bandit algorithm.

Implements contextual bandits that consider user context.

Example:
```python
    cb = ContextualBandit()
    
    arm = cb.select_arm(arms, context={"age": 25, "new_user": True})
    cb.update(arm, context, reward=1)
```
"""

from typing import Dict, List, Any

import numpy as np

from bufferiq.ml.experiments.bandits.thompson_sampling import BanditArm


class ContextualBandit:
    """
    Contextual bandit with linear models.

    Learns to select arms based on context features.

    Example:
```python
        cb = ContextualBandit(n_features=3)

        arms = [
            BanditArm("control", "Original"),
            BanditArm("treatment", "New")
        ]

        for _ in range(1000):
            context = {"age": 25, "tenure": 30, "segment": 1}
            arm = cb.select_arm(arms, context)
            reward = simulate_reward(arm, context)
            cb.update(arm, context, reward)
```
    """

    def __init__(self, n_features: int = 10, alpha: float = 1.0) -> None:
        """
        Initialize contextual bandit.

        Args:
            n_features: Number of context features
            alpha: Regularization parameter
        """
        self.n_features = n_features
        self.alpha = alpha

        # Store arm models (dict of variant_id -> model params)
        self.arm_models: Dict[str, Dict[str, np.ndarray]] = {}

    def _init_arm_model(self, variant_id: str) -> None:
        """
        Initialize model for arm.

        Args:
            variant_id: Variant ID
        """
        self.arm_models[variant_id] = {
            "A": np.eye(self.n_features) * self.alpha,  # Design matrix
            "b": np.zeros(self.n_features),  # Response vector
            "theta": np.zeros(self.n_features),  # Parameters
        }

    def _context_to_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Convert context to feature vector.

        Args:
            context: Context dictionary

        Returns:
            Feature vector
        """
        # Simple conversion: take numeric values
        features = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(float(value))
            else:
                features.append(0.0)

        # Pad or truncate to n_features
        features = features[: self.n_features]
        while len(features) < self.n_features:
            features.append(0.0)

        return np.array(features)

    def select_arm(
        self, arms: List[BanditArm], context: Dict[str, Any]
    ) -> BanditArm:
        """
        Select arm based on context.

        Args:
            arms: List of arms
            context: Context features

        Returns:
            Selected arm
        """
        if not arms:
            raise ValueError("No arms provided")

        # Convert context to features
        x = self._context_to_features(context)

        # Initialize models if needed
        for arm in arms:
            if arm.variant_id not in self.arm_models:
                self._init_arm_model(arm.variant_id)

        # Calculate UCB for each arm
        ucb_values = []
        for arm in arms:
            model = self.arm_models[arm.variant_id]

            # Expected reward
            expected_reward = np.dot(model["theta"], x)

            # Uncertainty
            A_inv = np.linalg.inv(model["A"])
            uncertainty = np.sqrt(np.dot(x, np.dot(A_inv, x)))

            # UCB
            ucb = expected_reward + self.alpha * uncertainty
            ucb_values.append(ucb)

        # Select arm with highest UCB
        best_idx = np.argmax(ucb_values)
        return arms[best_idx]

    def update(
        self, arm: BanditArm, context: Dict[str, Any], reward: float
    ) -> None:
        """
        Update arm model with observed reward.

        Args:
            arm: Selected arm
            context: Context features
            reward: Observed reward
        """
        # Convert context to features
        x = self._context_to_features(context)

        # Initialize model if needed
        if arm.variant_id not in self.arm_models:
            self._init_arm_model(arm.variant_id)

        model = self.arm_models[arm.variant_id]

        # Update design matrix and response vector
        model["A"] += np.outer(x, x)
        model["b"] += reward * x

        # Update parameters
        A_inv = np.linalg.inv(model["A"])
        model["theta"] = np.dot(A_inv, model["b"])

        # Update arm statistics
        arm.trials += 1
        if reward > 0:
            arm.successes += 1
        arm.mean_reward = arm.successes / arm.trials if arm.trials > 0 else 0.0