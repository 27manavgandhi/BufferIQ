"""
Interference detector.

Detects when treatment effects spill over to control group
(SUTVA violations).

Example:
```python
    detector = InterferenceDetector()
    
    result = detector.detect_interference(
        treatment_outcomes=[1, 1, 0, 1],
        control_outcomes=[1, 0, 0, 1],
        network_edges=[(0, 1), (2, 3)]
    )
```
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats


class InterferenceDetector:
    """
    Detect interference in experiments.

    Checks for spillover effects and SUTVA violations.

    Example:
```python
        detector = InterferenceDetector()

        result = detector.detect_interference(
            treatment_user_ids=["user1", "user2"],
            control_user_ids=["user3", "user4"],
            treatment_outcomes=[1, 1],
            control_outcomes=[0, 1],
            network_edges=[("user1", "user3")]
        )

        if result['has_interference']:
            print(f"Interference detected!")
            print(f"Affected users: {result['affected_count']}")
```
    """

    def detect_interference(
        self,
        treatment_user_ids: List[str],
        control_user_ids: List[str],
        treatment_outcomes: List[float],
        control_outcomes: List[float],
        network_edges: List[Tuple[str, str]],
    ) -> Dict[str, any]:
        """
        Detect interference based on network structure.

        Args:
            treatment_user_ids: Treatment user IDs
            control_user_ids: Control user IDs
            treatment_outcomes: Treatment outcomes
            control_outcomes: Control outcomes
            network_edges: Network connections

        Returns:
            Detection result
        """
        # Build user->variant mapping
        user_variant = {}
        for uid in treatment_user_ids:
            user_variant[uid] = "treatment"
        for uid in control_user_ids:
            user_variant[uid] = "control"

        # Build user->outcome mapping
        user_outcome = {}
        for uid, outcome in zip(treatment_user_ids, treatment_outcomes):
            user_outcome[uid] = outcome
        for uid, outcome in zip(control_user_ids, control_outcomes):
            user_outcome[uid] = outcome

        # Find cross-variant edges (treatment <-> control)
        cross_edges = []
        for u1, u2 in network_edges:
            if u1 in user_variant and u2 in user_variant:
                if user_variant[u1] != user_variant[u2]:
                    cross_edges.append((u1, u2))

        # Check if control users connected to treatment have different outcomes
        control_connected = []
        control_isolated = []

        for uid in control_user_ids:
            # Check if connected to treatment users
            connected = any(
                (uid == e[0] or uid == e[1]) for e in cross_edges
            )

            if connected:
                control_connected.append(user_outcome[uid])
            else:
                control_isolated.append(user_outcome[uid])

        # Test for difference
        has_interference = False
        p_value = 1.0

        if control_connected and control_isolated:
            # T-test
            if len(control_connected) > 1 and len(control_isolated) > 1:
                t_stat, p_value = stats.ttest_ind(
                    control_connected, control_isolated
                )
                has_interference = p_value < 0.05

        return {
            "has_interference": has_interference,
            "cross_edges_count": len(cross_edges),
            "control_connected_count": len(control_connected),
            "control_isolated_count": len(control_isolated),
            "p_value": float(p_value),
            "recommendation": (
                "Use cluster randomization"
                if has_interference
                else "SUTVA likely holds"
            ),
        }

    def calculate_exposure_probability(
        self,
        user_id: str,
        treatment_user_ids: List[str],
        network_edges: List[Tuple[str, str]],
    ) -> float:
        """
        Calculate probability of treatment exposure through network.

        Args:
            user_id: User ID
            treatment_user_ids: Treatment user IDs
            network_edges: Network edges

        Returns:
            Exposure probability (0-1)
        """
        # Count treatment neighbors
        treatment_neighbors = 0
        total_neighbors = 0

        for u1, u2 in network_edges:
            if u1 == user_id:
                total_neighbors += 1
                if u2 in treatment_user_ids:
                    treatment_neighbors += 1
            elif u2 == user_id:
                total_neighbors += 1
                if u1 in treatment_user_ids:
                    treatment_neighbors += 1

        if total_neighbors == 0:
            return 0.0

        return treatment_neighbors / total_neighbors

    def detect_spillover(
        self,
        control_outcomes_near_treatment: List[float],
        control_outcomes_far_from_treatment: List[float],
    ) -> Dict[str, any]:
        """
        Detect spillover by comparing control outcomes.

        Args:
            control_outcomes_near_treatment: Control near treatment
            control_outcomes_far_from_treatment: Control far from treatment

        Returns:
            Spillover detection result
        """
        if (
            not control_outcomes_near_treatment
            or not control_outcomes_far_from_treatment
        ):
            return {"has_spillover": False, "reason": "insufficient_data"}

        # Test for difference
        t_stat, p_value = stats.ttest_ind(
            control_outcomes_near_treatment,
            control_outcomes_far_from_treatment,
        )

        has_spillover = p_value < 0.05

        mean_near = float(np.mean(control_outcomes_near_treatment))
        mean_far = float(np.mean(control_outcomes_far_from_treatment))

        return {
            "has_spillover": has_spillover,
            "p_value": float(p_value),
            "mean_near_treatment": mean_near,
            "mean_far_from_treatment": mean_far,
            "spillover_effect": mean_near - mean_far,
        }