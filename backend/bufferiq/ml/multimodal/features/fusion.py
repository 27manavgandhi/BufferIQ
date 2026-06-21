"""Feature fusion strategies."""

from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureFusion:
    """Fuse features from multiple modalities."""
    
    def __init__(self, strategy: str = "concatenate"):
        """
        Initialize feature fusion.
        
        Args:
            strategy: Fusion strategy (concatenate, average, weighted)
        """
        self.strategy = strategy
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fuse(
        self,
        feature_vectors: List[np.ndarray],
        weights: List[float] | None = None
    ) -> np.ndarray:
        """
        Fuse multiple feature vectors.
        
        Args:
            feature_vectors: List of feature vectors
            weights: Optional weights for weighted fusion
            
        Returns:
            Fused feature vector
        """
        if not feature_vectors:
            raise ValueError("At least one feature vector required")
        
        if self.strategy == "concatenate":
            return self._concatenate(feature_vectors)
        elif self.strategy == "average":
            return self._average(feature_vectors)
        elif self.strategy == "weighted":
            if weights is None:
                weights = [1.0] * len(feature_vectors)
            return self._weighted(feature_vectors, weights)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
    
    def _concatenate(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        """Concatenate feature vectors."""
        return np.concatenate(feature_vectors)
    
    def _average(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        """Average feature vectors (requires same dimensions)."""
        # Ensure all vectors have same shape
        shapes = [v.shape for v in feature_vectors]
        if len(set(shapes)) > 1:
            raise ValueError("All feature vectors must have same shape for averaging")
        
        return np.mean(feature_vectors, axis=0)
    
    def _weighted(
        self,
        feature_vectors: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """Weighted average of feature vectors."""
        # Ensure all vectors have same shape
        shapes = [v.shape for v in feature_vectors]
        if len(set(shapes)) > 1:
            raise ValueError("All feature vectors must have same shape for weighting")
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()
        
        # Weighted sum
        weighted_sum = sum(
            w * v for w, v in zip(weights_array, feature_vectors)
        )
        
        return weighted_sum
    
    def fit_scaler(self, features: np.ndarray) -> None:
        """
        Fit scaler on features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        self.scaler.fit(features)
        self._is_fitted = True
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted scaler.
        
        Args:
            features: Feature vector or matrix
            
        Returns:
            Normalized features
        """
        if not self._is_fitted:
            # Fit on first use
            if features.ndim == 1:
                self.fit_scaler(features.reshape(1, -1))
            else:
                self.fit_scaler(features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features)[0]
        else:
            return self.scaler.transform(features)