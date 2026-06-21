"""Feature validation."""

from typing import List, Tuple
import numpy as np


class FeatureValidator:
    """Validate feature vectors."""
    
    def __init__(self):
        """Initialize feature validator."""
        pass
    
    def validate(
        self,
        features: np.ndarray,
        expected_dim: int | None = None
    ) -> Tuple[bool, str]:
        """
        Validate feature vector.
        
        Args:
            features: Feature vector
            expected_dim: Expected dimensionality
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if not isinstance(features, np.ndarray):
            return False, "Features must be numpy array"
        
        # Check shape
        if features.ndim != 1:
            return False, f"Features must be 1D array, got {features.ndim}D"
        
        # Check dimensionality
        if expected_dim is not None and len(features) != expected_dim:
            return False, f"Expected {expected_dim} features, got {len(features)}"
        
        # Check for NaN
        if np.isnan(features).any():
            return False, "Features contain NaN values"
        
        # Check for Inf
        if np.isinf(features).any():
            return False, "Features contain Inf values"
        
        return True, "Valid"
    
    def validate_batch(
        self,
        feature_batch: np.ndarray,
        expected_dim: int | None = None
    ) -> Tuple[bool, str]:
        """
        Validate batch of feature vectors.
        
        Args:
            feature_batch: Batch of feature vectors (n_samples, n_features)
            expected_dim: Expected feature dimensionality
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if not isinstance(feature_batch, np.ndarray):
            return False, "Feature batch must be numpy array"
        
        # Check shape
        if feature_batch.ndim != 2:
            return False, f"Feature batch must be 2D array, got {feature_batch.ndim}D"
        
        # Check dimensionality
        if expected_dim is not None and feature_batch.shape[1] != expected_dim:
            return False, f"Expected {expected_dim} features, got {feature_batch.shape[1]}"
        
        # Check for NaN
        if np.isnan(feature_batch).any():
            return False, "Feature batch contains NaN values"
        
        # Check for Inf
        if np.isinf(feature_batch).any():
            return False, "Feature batch contains Inf values"
        
        return True, "Valid"
    
    def check_feature_range(
        self,
        features: np.ndarray,
        min_val: float = -10.0,
        max_val: float = 10.0
    ) -> Tuple[bool, str]:
        """
        Check if features are within expected range.
        
        Args:
            features: Feature vector
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if features.min() < min_val:
            return False, f"Features contain values below {min_val}"
        
        if features.max() > max_val:
            return False, f"Features contain values above {max_val}"
        
        return True, "Valid"