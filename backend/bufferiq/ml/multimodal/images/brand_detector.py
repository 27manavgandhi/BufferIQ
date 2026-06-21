"""Brand element detection in images."""

from typing import List
import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.exceptions import AnalysisError


class BrandDetector:
    """Detect brand elements in images."""
    
    def __init__(self):
        """Initialize brand detector."""
        # In production, load brand logo templates or use trained model
        self.known_brands = [
            "company_logo",
            "brand_watermark",
            "trademark_symbol",
        ]
    
    def detect(self, image: Image.Image) -> List[str]:
        """
        Detect brand elements in image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected brand elements
            
        Raises:
            AnalysisError: If detection fails
        """
        try:
            detected = []
            
            # Simulate brand detection
            # In production, use template matching or trained model
            
            # Check for watermark-like patterns (corners)
            has_corner_pattern = self._check_corner_patterns(image)
            if has_corner_pattern:
                detected.append("brand_watermark")
            
            # Check for logo-like features
            has_logo_features = self._check_logo_features(image)
            if has_logo_features:
                detected.append("company_logo")
            
            return detected
            
        except Exception as e:
            raise AnalysisError(f"Brand detection failed: {str(e)}")
    
    def _check_corner_patterns(self, image: Image.Image) -> bool:
        """
        Check for patterns in image corners (common watermark location).
        
        Args:
            image: PIL Image
            
        Returns:
            True if corner patterns detected
        """
        width, height = image.size
        corner_size = min(width, height) // 10
        
        # Check bottom-right corner (most common watermark location)
        corner = image.crop((
            width - corner_size,
            height - corner_size,
            width,
            height
        ))
        
        # Convert to grayscale and check variance
        gray = np.array(corner.convert('L'))
        variance = np.var(gray)
        
        # High variance suggests pattern/text
        return variance > 500
    
    def _check_logo_features(self, image: Image.Image) -> bool:
        """
        Check for logo-like features.
        
        Args:
            image: PIL Image
            
        Returns:
            True if logo features detected
        """
        # Resize for faster processing
        img_small = image.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = np.array(img_small.convert('L'))
        
        # Check for high contrast regions (typical in logos)
        std_dev = np.std(gray)
        
        return std_dev > 40