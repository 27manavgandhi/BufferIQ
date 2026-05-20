"""
Stratifier for experiment assignment.

Handles stratified randomization to ensure balanced
variant assignment across strata.

Example:
```python
    stratifier = Stratifier()
    
    strata = stratifier.create_strata(
        users=users,
        stratification_key="user_type"
    )
    
    # Each stratum gets balanced assignment
```
"""

from typing import Any, Dict, List

import numpy as np


class Stratifier:
    """
    Stratified randomization for experiments.

    Ensures balanced variant assignment within strata
    to reduce variance and improve statistical power.

    Example:
```python
        stratifier = Stratifier()

        strata = stratifier.create_strata(
            users=user_list,
            stratification_key="segment"
        )

        for stratum_key, stratum_users in strata.items():
            print(f"{stratum_key}: {len(stratum_users)} users")
```
    """

    def create_strata(
        self, users: List[Dict[str, Any]], stratification_key: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create strata from user list.

        Args:
            users: List of users
            stratification_key: Key to stratify by

        Returns:
            Dictionary mapping stratum key to users
        """
        strata: Dict[str, List[Dict[str, Any]]] = {}

        for user in users:
            stratum_value = user.get(stratification_key, "unknown")
            stratum_key = str(stratum_value)

            if stratum_key not in strata:
                strata[stratum_key] = []

            strata[stratum_key].append(user)

        return strata

    def validate_balance(
        self,
        assignments: Dict[str, List[str]],
        stratification_key: str,
        users: List[Dict[str, Any]],
        tolerance: float = 0.1,
    ) -> bool:
        """
        Validate that assignments are balanced within strata.

        Args:
            assignments: Variant assignments {variant_id: [user_ids]}
            stratification_key: Stratification key
            users: User data
            tolerance: Maximum allowed imbalance (0-1)

        Returns:
            True if balanced, False otherwise
        """
        # Create user lookup
        user_lookup = {u["user_id"]: u for u in users}

        # Get strata
        strata = self.create_strata(users, stratification_key)

        # Check balance in each stratum
        for stratum_key, stratum_users in strata.items():
            stratum_user_ids = {u["user_id"] for u in stratum_users}

            # Count assignments per variant in this stratum
            variant_counts = {}
            for variant_id, user_ids in assignments.items():
                count = len(set(user_ids) & stratum_user_ids)
                variant_counts[variant_id] = count

            # Check balance
            total = sum(variant_counts.values())
            if total == 0:
                continue

            expected_per_variant = total / len(variant_counts)

            for count in variant_counts.values():
                actual_proportion = count / total
                expected_proportion = expected_per_variant / total
                imbalance = abs(actual_proportion - expected_proportion)

                if imbalance > tolerance:
                    return False

        return True