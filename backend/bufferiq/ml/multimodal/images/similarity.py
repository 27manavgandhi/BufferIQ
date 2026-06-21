"""Image similarity computation."""

from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import cosine


class ImageSimilarityCalculator:
    """Calculate similarity between images using embeddings."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        pass
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First image embedding
            embedding2: Second image embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        similarity = 1.0 - cosine(embedding1, embedding2)
        return float(max(0.0, min(similarity, 1.0)))
    
    def find_similar_images(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar images.
        
        Args:
            query_embedding: Query image embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]