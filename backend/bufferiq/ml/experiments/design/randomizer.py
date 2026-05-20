"""
Randomizer for experiment assignment.

Provides different randomization strategies for
assigning users to variants.

Example:
```python
    randomizer = Randomizer(seed=42)
    
    variant = randomizer.simple_random(
        variants=variants,
        user_id="user123"
    )
```
"""

from typing import List, Optional

import hashlib
import numpy as np

from bufferiq.ml.experiments.design.designer import Variant


class Randomizer:
    """
    Randomization strategies for experiment assignment.

    Provides multiple strategies:
    - Simple random assignment
    - Deterministic hash-based
    - Blocked randomization
    - Stratified randomization

    Example:
```python
        randomizer = Randomizer(seed=42)

        variant = randomizer.hash_based(
            variants=variants,
            user_id="user123",
            experiment_id="exp_001"
        )
```
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize randomizer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def simple_random(self, variants: List[Variant]) -> Variant:
        """
        Simple random assignment.

        Args:
            variants: List of variants

        Returns:
            Randomly selected variant
        """
        # Get traffic allocations
        allocations = [v.traffic_allocation for v in variants]

        # Random selection
        idx = np.random.choice(len(variants), p=allocations)

        return variants[idx]

    def hash_based(
        self, variants: List[Variant], user_id: str, experiment_id: str
    ) -> Variant:
        """
        Deterministic hash-based assignment.

        Same user + experiment always gets same variant.

        Args:
            variants: List of variants
            user_id: User identifier
            experiment_id: Experiment identifier

        Returns:
            Assigned variant
        """
        # Generate hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-1
        bucket = (hash_value % 10000) / 10000.0

        # Assign based on traffic allocation
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_allocation
            if bucket < cumulative:
                return variant

        # Fallback to last variant
        return variants[-1]

    def blocked_random(
        self, variants: List[Variant], block_size: int = 10
    ) -> List[Variant]:
        """
        Blocked randomization.

        Creates balanced blocks of assignments.

        Args:
            variants: List of variants
            block_size: Size of each block

        Returns:
            List of variant assignments
        """
        # Calculate how many of each variant per block
        allocations = [v.traffic_allocation for v in variants]
        counts = [int(alloc * block_size) for alloc in allocations]

        # Adjust for rounding
        while sum(counts) < block_size:
            idx = np.argmax(allocations)
            counts[idx] += 1

        # Create block
        block = []
        for variant, count in zip(variants, counts):
            block.extend([variant] * count)

        # Shuffle block
        np.random.shuffle(block)

        return block