"""Image preprocessing utilities."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io

from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class ImagePreprocessor:
    """Preprocess images for analysis."""
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        max_size: int = 2048,
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target size for resizing (width, height)
            max_size: Maximum dimension size
        """
        self.target_size = target_size
        self.max_size = max_size
    
    def load_image(self, image_source: str | bytes | Image.Image) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            image_source: Image path, bytes, or PIL Image
            
        Returns:
            PIL Image object
            
        Raises:
            MediaProcessingError: If image loading fails
        """
        try:
            if isinstance(image_source, Image.Image):
                return image_source
            elif isinstance(image_source, bytes):
                return Image.open(io.BytesIO(image_source))
            elif isinstance(image_source, str):
                return Image.open(image_source)
            else:
                raise MediaProcessingError(f"Unsupported image source type: {type(image_source)}")
        except Exception as e:
            raise MediaProcessingError(f"Failed to load image: {str(e)}")
    
    def resize(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image
            
        Returns:
            Resized image
        """
        if self.target_size:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Resize if larger than max_size
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_size = (int(width * scale), int(height * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def normalize(self, image: Image.Image) -> np.ndarray:
        """
        Normalize image to numpy array.
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized numpy array
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
    
    def preprocess(self, image_source: str | bytes | Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Full preprocessing pipeline.
        
        Args:
            image_source: Image source
            
        Returns:
            Tuple of (PIL Image, normalized array)
        """
        image = self.load_image(image_source)
        image = self.resize(image)
        normalized = self.normalize(image)
        
        return image, normalized