"""Video embedding generation."""

from typing import List
import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.types import KeyFrame
from bufferiq.ml.multimodal.images.embeddings import ImageEmbeddingGenerator
from bufferiq.ml.multimodal.exceptions import FeatureExtractionError


class VideoEmbeddingGenerator:
    """Generate video embeddings from key frames."""
    
    def __init__(self):
        """Initialize video embedding generator."""
        self.image_embedder = ImageEmbeddingGenerator()
    
    def generate(self, keyframes: List[KeyFrame]) -> np.ndarray:
        """
        Generate video embedding from key frames.
        
        Args:
            keyframes: List of key frames
            
        Returns:
            Video embedding vector
            
        Raises:
            FeatureExtractionError: If generation fails
        """
        try:
            if not keyframes:
                # Return zero embedding if no keyframes
                dim = self.image_embedder.get_embedding_dim()
                return np.zeros(dim)
            
            # Generate embeddings for each keyframe
            frame_embeddings = []
            
            for keyframe in keyframes:
                # Load frame image (in production)
                # For now, create dummy embedding
                dim = self.image_embedder.get_embedding_dim()
                embedding = np.random.randn(dim)
                embedding = embedding / np.linalg.norm(embedding)
                frame_embeddings.append(embedding)
            
            # Aggregate embeddings (average)
            video_embedding = np.mean(frame_embeddings, axis=0)
            
            # Normalize
            video_embedding = video_embedding / np.linalg.norm(video_embedding)
            
            return video_embedding
            
        except Exception as e:
            raise FeatureExtractionError(f"Video embedding generation failed: {str(e)}")