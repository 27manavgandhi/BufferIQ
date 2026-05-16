"""Tests for consistency metrics."""

import pytest
import math
from bufferiq.ml.voice.consistency.metrics import ConsistencyMetrics


class TestConsistencyMetrics:
    """Test consistency metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return ConsistencyMetrics()
    
    def test_cosine_similarity_identical_vectors(self, metrics):
        """Test cosine similarity for identical vectors."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        
        similarity = metrics.cosine_similarity(vec, vec)
        
        assert similarity == pytest.approx(1.0)
    
    def test_cosine_similarity_orthogonal_vectors(self, metrics):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        
        similarity = metrics.cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(0.0)
    
    def test_cosine_similarity_opposite_vectors(self, metrics):
        """Test cosine similarity for opposite vectors."""
        vec1 = {"a": 1.0}
        vec2 = {"a": -1.0}
        
        similarity = metrics.cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(-1.0)
    
    def test_cosine_similarity_empty_vectors(self, metrics):
        """Test cosine similarity with empty vectors."""
        similarity = metrics.cosine_similarity({}, {})
        
        assert similarity == 0.0
    
    def test_cosine_similarity_one_empty(self, metrics):
        """Test cosine similarity with one empty vector."""
        vec = {"a": 1.0}
        
        similarity = metrics.cosine_similarity(vec, {})
        
        assert similarity == 0.0
    
    def test_euclidean_distance_identical(self, metrics):
        """Test Euclidean distance for identical vectors."""
        vec = {"a": 1.0, "b": 2.0}
        
        distance = metrics.euclidean_distance(vec, vec)
        
        assert distance == pytest.approx(0.0)
    
    def test_euclidean_distance_perpendicular(self, metrics):
        """Test Euclidean distance for perpendicular unit vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        
        distance = metrics.euclidean_distance(vec1, vec2)
        
        # sqrt(1^2 + 1^2) = sqrt(2)
        assert distance == pytest.approx(math.sqrt(2))
    
    def test_euclidean_distance_empty_vectors(self, metrics):
        """Test Euclidean distance with empty vectors."""
        distance = metrics.euclidean_distance({}, {})
        
        assert distance == float('inf')
    
    def test_kl_divergence_identical_distributions(self, metrics):
        """Test KL divergence for identical distributions."""
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        
        divergence = metrics.kl_divergence(dist, dist)
        
        assert divergence == pytest.approx(0.0, abs=0.01)
    
    def test_kl_divergence_different_distributions(self, metrics):
        """Test KL divergence for different distributions."""
        dist1 = {"a": 0.7, "b": 0.3}
        dist2 = {"a": 0.3, "b": 0.7}
        
        divergence = metrics.kl_divergence(dist1, dist2)
        
        assert divergence > 0
    
    def test_kl_divergence_empty_distributions(self, metrics):
        """Test KL divergence with empty distributions."""
        divergence = metrics.kl_divergence({}, {})
        
        assert divergence == float('inf')
    
    def test_kl_divergence_normalization(self, metrics):
        """Test KL divergence normalizes distributions."""
        # Non-normalized distributions
        dist1 = {"a": 5.0, "b": 3.0}
        dist2 = {"a": 3.0, "b": 5.0}
        
        divergence = metrics.kl_divergence(dist1, dist2)
        
        # Should still work and be > 0
        assert divergence > 0
    
    def test_manhattan_distance_identical(self, metrics):
        """Test Manhattan distance for identical vectors."""
        vec = {"a": 1.0, "b": 2.0}
        
        distance = metrics.manhattan_distance(vec, vec)
        
        assert distance == pytest.approx(0.0)
    
    def test_manhattan_distance_orthogonal(self, metrics):
        """Test Manhattan distance for orthogonal vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        
        distance = metrics.manhattan_distance(vec1, vec2)
        
        # |1-0| + |0-1| = 2
        assert distance == pytest.approx(2.0)
    
    def test_manhattan_distance_empty_vectors(self, metrics):
        """Test Manhattan distance with empty vectors."""
        distance = metrics.manhattan_distance({}, {})
        
        assert distance == float('inf')
    
    def test_cosine_similarity_partial_overlap(self, metrics):
        """Test cosine similarity with partial key overlap."""
        vec1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec2 = {"b": 2.0, "c": 3.0, "d": 4.0}
        
        similarity = metrics.cosine_similarity(vec1, vec2)
        
        # Should handle partial overlap
        assert 0 <= similarity <= 1.0
    
    def test_euclidean_distance_partial_overlap(self, metrics):
        """Test Euclidean distance with partial key overlap."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"b": 3.0, "c": 4.0}
        
        distance = metrics.euclidean_distance(vec1, vec2)
        
        # Should handle partial overlap
        assert distance > 0
    
    def test_kl_divergence_missing_keys(self, metrics):
        """Test KL divergence handles missing keys."""
        dist1 = {"a": 0.6, "b": 0.4}
        dist2 = {"a": 0.5, "c": 0.5}
        
        divergence = metrics.kl_divergence(dist1, dist2)
        
        # Should handle missing keys gracefully
        assert divergence >= 0