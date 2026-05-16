"""
Voice profile comparison utilities.

Compares voice profiles across different brands.
"""

from typing import Dict, List
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile
from bufferiq.ml.voice.consistency.metrics import ConsistencyMetrics

logger = logging.getLogger(__name__)


class VoiceComparator:
    """
    Compare voice profiles across brands.
    
    Analyzes similarities and differences between
    different brand voices.
    
    Example:
```python
        comparator = VoiceComparator()
        similarity = comparator.compare_profiles(profile_a, profile_b)
        print(f"Similarity: {similarity:.2f}")
        
        differences = comparator.identify_differences(profile_a, profile_b)
```
    """
    
    def __init__(self):
        """Initialize voice comparator."""
        self.metrics = ConsistencyMetrics()
    
    def compare_profiles(
        self, profile_a: VoiceProfile, profile_b: VoiceProfile
    ) -> float:
        """
        Compare two voice profiles for similarity.
        
        Args:
            profile_a: First profile
            profile_b: Second profile
        
        Returns:
            Similarity score (0-1)
        """
        # Combine all fingerprints
        fp_a = {
            **profile_a.lexical_fingerprint,
            **profile_a.syntactic_fingerprint,
            **profile_a.stylistic_fingerprint,
        }
        
        fp_b = {
            **profile_b.lexical_fingerprint,
            **profile_b.syntactic_fingerprint,
            **profile_b.stylistic_fingerprint,
        }
        
        # Calculate cosine similarity
        similarity = self.metrics.cosine_similarity(fp_a, fp_b)
        
        logger.info(
            f"Compared {profile_a.brand_id} vs {profile_b.brand_id}: "
            f"similarity = {similarity:.3f}"
        )
        
        return similarity
    
    def identify_differences(
        self, profile_a: VoiceProfile, profile_b: VoiceProfile
    ) -> Dict[str, float]:
        """
        Identify key differences between profiles.
        
        Args:
            profile_a: First profile
            profile_b: Second profile
        
        Returns:
            Dictionary of dimension -> difference
        """
        differences = {}
        
        # Compare lexical dimensions
        for key in profile_a.lexical_fingerprint:
            val_a = profile_a.lexical_fingerprint.get(key, 0)
            val_b = profile_b.lexical_fingerprint.get(key, 0)
            diff = abs(val_a - val_b)
            if diff > 0.1:  # Threshold
                differences[f"lexical_{key}"] = diff
        
        # Compare stylistic dimensions
        for key in profile_a.stylistic_fingerprint:
            val_a = profile_a.stylistic_fingerprint.get(key, 0)
            val_b = profile_b.stylistic_fingerprint.get(key, 0)
            diff = abs(val_a - val_b)
            if diff > 5.0:  # Threshold
                differences[f"stylistic_{key}"] = diff
        
        return differences
    
    def rank_by_similarity(
        self, target: VoiceProfile, candidates: List[VoiceProfile]
    ) -> List[tuple]:
        """
        Rank profiles by similarity to target.
        
        Args:
            target: Target profile
            candidates: List of candidate profiles
        
        Returns:
            List of (profile, similarity) tuples, sorted by similarity
        """
        similarities = []
        
        for candidate in candidates:
            if candidate.profile_id != target.profile_id:
                similarity = self.compare_profiles(target, candidate)
                similarities.append((candidate, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities