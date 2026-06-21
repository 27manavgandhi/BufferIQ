"""Image embedding generation using CLIP."""

from typing import Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from bufferiq.ml.multimodal.exceptions import FeatureExtractionError


class ImageEmbeddingGenerator:
    """Generate image embeddings using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: CLIP model name
        """
        self.model_name = model_name
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
    
    @property
    def model(self) -> CLIPModel:
        """Lazy load model."""
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model
    
    @property
    def processor(self) -> CLIPProcessor:
        """Lazy load processor."""
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor
    
    def generate(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for image.
        
        Args:
            image: PIL Image
            
        Returns:
            Image embedding vector
            
        Raises:
            FeatureExtractionError: If generation fails
        """
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy()[0]
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            raise FeatureExtractionError(f"Embedding generation failed: {str(e)}")
    
    def get_embedding_dim(self) -> int:
        """
        Get embedding dimensionality.
        
        Returns:
            Embedding dimension
        """
        return self.model.config.projection_dim