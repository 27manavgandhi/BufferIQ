"""
Sequential Probability Ratio Test (SPRT).

Implements sequential testing with early stopping.

Example:
```python
    tester = SequentialTester(alpha=0.05, beta=0.20)
    
    result = tester.test(
        control_successes=50,
        control_trials=1000,
        treatment_successes=60,
        treatment_trials=1000
    )
    
    if result['decision'] == 'stop':
        print(f"Early stop: {result['conclusion']}")
```
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
from scipy import stats


@dataclass
class SPRTResult:
    """SPRT test result."""

    decision: Literal["continue", "stop"]
    conclusion: Optional[str]  # "treatment_wins", "control_wins", None
    log_likelihood_ratio: float
    lower_boundary: float
    upper_boundary: float
    n_control: int
    n_treatment: int


class SequentialTester:
    """
    Sequential Probability Ratio Test.

    Allows early stopping when sufficient evidence is collected.

    Example:
```python
        tester = SequentialTester(alpha=0.05, beta=0.20)

        result = tester.test(
            control_successes=50,
            control_trials=1000,
            treatment_successes=60,
            treatment_trials=1000
        )

        if result.decision == "stop":
            print(f"Stop early: {result.conclusion}")
        else:
            print("Continue collecting data")
```
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.20) -> None:
        """
        Initialize sequential tester.

        Args:
            alpha: Type I error rate
            beta: Type II error rate (1 - power)
        """
        self.alpha = alpha
        self.beta = beta

        # Calculate boundaries
        self.upper_boundary = np.log((1 - beta) / alpha)
        self.lower_boundary = np.log(beta / (1 - alpha))

    def test(
        self,
        control_successes: int,
        control_trials: int,
        treatment_successes: int,
        treatment_trials: int,
        mde: float = 0.10,
    ) -> SPRTResult:
        """
        Perform SPRT test.

        Args:
            control_successes: Control successes
            control_trials: Control trials
            treatment_successes: Treatment successes
            treatment_trials: Treatment trials
            mde: Minimum detectable effect

        Returns:
            SPRT result
        """
        # Calculate rates
        p_control = control_successes / control_trials if control_trials > 0 else 0.0
        p_treatment = (
            treatment_successes / treatment_trials if treatment_trials > 0 else 0.0
        )

        # Null and alternative hypotheses
        p0 = p_control  # Null: no difference
        p1 = p_control * (1 + mde)  # Alternative: MDE improvement

        # Log-likelihood ratio
        llr = self._calculate_llr(
            treatment_successes,
            treatment_trials - treatment_successes,
            p0,
            p1,
        )

        # Make decision
        if llr >= self.upper_boundary:
            decision = "stop"
            conclusion = "treatment_wins"
        elif llr <= self.lower_boundary:
            decision = "stop"
            conclusion = "control_wins"
        else:
            decision = "continue"
            conclusion = None

        return SPRTResult(
            decision=decision,
            conclusion=conclusion,
            log_likelihood_ratio=llr,
            lower_boundary=self.lower_boundary,
            upper_boundary=self.upper_boundary,
            n_control=control_trials,
            n_treatment=treatment_trials,
        )

    def _calculate_llr(
        self, successes: int, failures: int, p0: float, p1: float
    ) -> float:
        """
        Calculate log-likelihood ratio.

        Args:
            successes: Number of successes
            failures: Number of failures
            p0: Null hypothesis probability
            p1: Alternative hypothesis probability

        Returns:
            Log-likelihood ratio
        """
        if p0 <= 0 or p0 >= 1 or p1 <= 0 or p1 >= 1:
            return 0.0

        llr = (
            successes * np.log(p1 / p0)
            + failures * np.log((1 - p1) / (1 - p0))
        )

        return float(llr)

    def calculate_stopping_time(
        self, p0: float, p1: float, sample_size: int
    ) -> int:
        """
        Estimate expected stopping time.

        Args:
            p0: Null probability
            p1: Alternative probability
            sample_size: Maximum sample size

        Returns:
            Expected stopping time
        """
        # Simplified estimate
        variance = p0 * (1 - p0)
        effect_size = p1 - p0

        if effect_size == 0:
            return sample_size

        est_time = int(
            (self.upper_boundary - self.lower_boundary) / (effect_size**2 / variance)
        )

        return min(est_time, sample_size)