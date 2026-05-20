"""
Group sequential testing.

Implements group sequential designs with multiple interim analyses.

Example:
```python
    tester = GroupSequentialTester(num_looks=3)
    
    result = tester.test(
        control_data=control,
        treatment_data=treatment,
        look_number=2
    )
```
"""

from typing import Dict, List

import numpy as np
from scipy import stats


class GroupSequentialTester:
    """
    Group sequential testing.

    Allows multiple interim looks with controlled Type I error.

    Example:
```python
        tester = GroupSequentialTester(
            num_looks=4,
            alpha=0.05
        )

        # At second interim look
        result = tester.test_at_look(
            control_data=control,
            treatment_data=treatment,
            look_number=2
        )

        if result['stop_for_efficacy']:
            print("Stop - treatment wins")
```
    """

    def __init__(
        self, num_looks: int = 3, alpha: float = 0.05, spending_function: str = "obrien_fleming"
    ) -> None:
        """
        Initialize group sequential tester.

        Args:
            num_looks: Number of planned interim looks
            alpha: Overall Type I error rate
            spending_function: Alpha spending function
        """
        self.num_looks = num_looks
        self.alpha = alpha
        self.spending_function = spending_function

        # Calculate boundaries
        self.boundaries = self._calculate_boundaries()

    def _calculate_boundaries(self) -> List[float]:
        """
        Calculate critical values for each look.

        Returns:
            List of critical z-values
        """
        if self.spending_function == "obrien_fleming":
            # O'Brien-Fleming boundaries
            boundaries = []
            for k in range(1, self.num_looks + 1):
                # Information fraction
                t = k / self.num_looks

                # Critical value
                z_crit = stats.norm.ppf(1 - self.alpha / (2 * np.sqrt(t)))
                boundaries.append(z_crit)

            return boundaries

        elif self.spending_function == "pocock":
            # Pocock boundaries (constant)
            z_crit = stats.norm.ppf(1 - self.alpha / (2 * self.num_looks))
            return [z_crit] * self.num_looks

        else:
            raise ValueError(f"Unknown spending function: {self.spending_function}")

    def test_at_look(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        look_number: int,
    ) -> Dict[str, any]:
        """
        Perform test at interim look.

        Args:
            control_data: Control data
            treatment_data: Treatment data
            look_number: Current look number (1-indexed)

        Returns:
            Test result
        """
        if look_number < 1 or look_number > self.num_looks:
            raise ValueError(f"Invalid look number: {look_number}")

        # Calculate test statistic
        mean_c = float(np.mean(control_data))
        mean_t = float(np.mean(treatment_data))

        var_c = float(np.var(control_data, ddof=1))
        var_t = float(np.var(treatment_data, ddof=1))

        n_c = len(control_data)
        n_t = len(treatment_data)

        # Pooled standard error
        se = np.sqrt(var_c / n_c + var_t / n_t)

        # Z-statistic
        z_stat = (mean_t - mean_c) / se if se > 0 else 0.0

        # Get critical value for this look
        z_crit = self.boundaries[look_number - 1]

        # Decision
        stop_for_efficacy = abs(z_stat) > z_crit
        stop_for_futility = abs(z_stat) < z_crit * 0.1  # Simplified futility

        return {
            "look_number": look_number,
            "z_statistic": z_stat,
            "z_critical": z_crit,
            "stop_for_efficacy": stop_for_efficacy,
            "stop_for_futility": stop_for_futility,
            "continue": not (stop_for_efficacy or stop_for_futility),
        }