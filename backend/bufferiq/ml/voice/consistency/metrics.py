"""
Consistency metrics calculations.

Implements various similarity and distance metrics
for voice consistency measurement.
"""

from typing import Dict
import math


class ConsistencyMetrics:
    """
    Calculate consistency metrics.
    
    Provides various similarity and distance calculations
    for voice consistency measurement.
    
    Example:
```python
        metrics = ConsistencyMetrics()
        similarity = metrics.cosine_similarity(vec1, vec2)
        divergence = metrics.kl_divergence(dist1, dist2)
```
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def cosine_similarity(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
        
        Returns:
            Cosine similarity (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Get all keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product
        dot_product = sum(
            vec1.get(key, 0) * vec2.get(key, 0) for key in all_keys
        )
        
        # Calculate magnitudes
        magnitude1 = sum(v ** 2 for v in vec1.values()) ** 0.5
        magnitude2 = sum(v ** 2 for v in vec2.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def euclidean_distance(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """
        Calculate Euclidean distance between two feature vectors.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
        
        Returns:
            Euclidean distance (0-∞)
        """
        if not vec1 or not vec2:
            return float('inf')
        
        # Get all keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate squared differences
        squared_diff = sum(
            (vec1.get(key, 0) - vec2.get(key, 0)) ** 2 for key in all_keys
        )
        
        return squared_diff ** 0.5
    
    def kl_divergence(
        self, dist1: Dict[str, float], dist2: Dict[str, float]
    ) -> float:
        """
        Calculate Kullback-Leibler divergence (simplified).
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
        
        Returns:
            KL divergence (0-∞, lower = more similar)
        """
        if not dist1 or not dist2:
            return float('inf')
        
        # Normalize distributions
        total1 = sum(abs(v) for v in dist1.values())
        total2 = sum(abs(v) for v in dist2.values())
        
        if total1 == 0 or total2 == 0:
            return float('inf')
        
        norm_dist1 = {k: abs(v) / total1 for k, v in dist1.items()}
        norm_dist2 = {k: abs(v) / total2 for k, v in dist2.items()}
        
        # Calculate KL divergence
        kl_div = 0.0
        epsilon = 1e-10  # Small value to avoid log(0)
        
        for key in norm_dist1.keys():
            p = norm_dist1[key] + epsilon
            q = norm_dist2.get(key, epsilon)
            kl_div += p * math.log(p / q)
        
        return kl_div
    
    def manhattan_distance(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """
        Calculate Manhattan (L1) distance.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
        
        Returns:
            Manhattan distance (0-∞)
        """
        if not vec1 or not vec2:
            return float('inf')
        
        # Get all keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate sum of absolute differences
        return sum(
            abs(vec1.get(key, 0) - vec2.get(key, 0)) for key in all_keys
        )