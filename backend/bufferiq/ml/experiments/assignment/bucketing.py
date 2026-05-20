"""
Hash-based bucketing.

Deterministic variant assignment using consistent hashing.

Example:
```python
    bucketing = HashBucketing()
    
    variant = bucketing.assign_variant(
        experiment_id="exp_001",
        user_id="user123",
        variants=variants
    )
```
"""

import hashlib
from typing import List

from bufferiq.ml.experiments.design.designer import Variant


class HashBucketing:
    """
    Hash-based bucketing for deterministic assignment.

    Uses MD5 hashing to assign users to variants consistently.

    Example:
```python
        bucketing = HashBucketing()

        variant = bucketing.assign_variant(
            experiment_id="exp_001",
            user_id="user123",
            variants=variants
        )

        # Same inputs always return same variant
        variant2 = bucketing.assign_variant("exp_001", "user123", variants)
        assert variant.id == variant2.id
```
    """

    def assign_variant(
        self, experiment_id: str, user_id: str, variants: List[Variant]
    ) -> Variant:
        """
        Assign variant using consistent hashing.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            variants: List of variants

        Returns:
            Assigned variant
        """
        # Generate hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-1
        bucket = (hash_value % 10000) / 10000.0

        # Assign to variant based on traffic allocation
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_allocation
            if bucket < cumulative:
                return variant

        # Fallback to last variant
        return variants[-1]

    def get_bucket(self, experiment_id: str, user_id: str) -> float:
        """
        Get bucket value for user.

        Args:
            experiment_id: Experiment ID
            user_id: User ID

        Returns:
            Bucket value (0-1)
        """
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return (hash_value % 10000) / 10000.0