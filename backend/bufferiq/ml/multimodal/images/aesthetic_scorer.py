"""Aesthetic quality scoring for images."""

import numpy as np
from PIL import Image, ImageStat

from bufferiq.ml.multimodal.exceptions import AnalysisError


class AestheticScorer:
    """Score aesthetic quality of images."""
    
    def __init__(self):
        """Initialize aesthetic scorer."""
        pass
    
    def score(self, image: Image.Image) -> float:
        """
        Score aesthetic quality.
        
        Args:
            image: PIL Image
            
        Returns:
            Aesthetic score (0-100)
            
        Raises:
            AnalysisError: If scoring fails
        """
        try:
            # Multiple aesthetic factors
            sharpness = self._score_sharpness(image)
            contrast = self._score_contrast(image)
            saturation = self._score_saturation(image)
            brightness = self._score_brightness(image)
            complexity = self._score_complexity(image)
            
            # Weighted combination
            aesthetic_score = (
                sharpness * 0.25 +
                contrast * 0.20 +
                saturation * 0.20 +
                brightness * 0.15 +
                complexity * 0.20
            )
            
            return float(aesthetic_score * 100)
            
        except Exception as e:
            raise AnalysisError(f"Aesthetic scoring failed: {str(e)}")
    
    def _score_sharpness(self, image: Image.Image) -> float:
        """
        Score image sharpness.
        
        Args:
            image: PIL Image
            
        Returns:
            Sharpness score (0-1)
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'), dtype=np.float32)
        
        # Calculate Laplacian variance (measure of sharpness)
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        
        # Apply filter
        from scipy.ndimage import convolve
        filtered = convolve(gray, laplacian)
        variance = np.var(filtered)
        
        # Normalize to 0-1
        score = min(variance / 1000.0, 1.0)
        return float(score)
    
    def _score_contrast(self, image: Image.Image) -> float:
        """
        Score image contrast.
        
        Args:
            image: PIL Image
            
        Returns:
            Contrast score (0-1)
        """
        # Convert to grayscale
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        
        # Standard deviation as measure of contrast
        std_dev = stat.stddev[0]
        
        # Normalize (std_dev typically 0-70 for good contrast)
        score = min(std_dev / 70.0, 1.0)
        return float(score)
    
    def _score_saturation(self, image: Image.Image) -> float:
        """
        Score color saturation.
        
        Args:
            image: PIL Image
            
        Returns:
            Saturation score (0-1)
        """
        # Convert to HSV
        hsv = image.convert('HSV')
        stat = ImageStat.Stat(hsv)
        
        # Get saturation channel (index 1)
        saturation = stat.mean[1]
        
        # Normalize (saturation 0-255)
        score = saturation / 255.0
        return float(score)
    
    def _score_brightness(self, image: Image.Image) -> float:
        """
        Score image brightness.
        
        Args:
            image: PIL Image
            
        Returns:
            Brightness score (0-1)
        """
        # Convert to grayscale
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        
        # Get mean brightness
        brightness = stat.mean[0]
        
        # Optimal brightness is around 127 (middle)
        # Score based on distance from optimal
        distance = abs(brightness - 127)
        score = 1.0 - (distance / 127.0)
        
        return float(max(0.0, score))
    
    def _score_complexity(self, image: Image.Image) -> float:
        """
        Score visual complexity.
        
        Args:
            image: PIL Image
            
        Returns:
            Complexity score (0-1)
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'), dtype=np.float32)
        
        # Calculate edge density
        dy = np.abs(np.diff(gray, axis=0))
        dx = np.abs(np.diff(gray, axis=1))
        edge_density = (dy.mean() + dx.mean()) / 255.0
        
        # Optimal complexity is moderate (not too simple, not too busy)
        # Score based on being in sweet spot (0.2-0.4)
        if edge_density < 0.2:
            score = edge_density / 0.2
        elif edge_density > 0.4:
            score = 1.0 - (edge_density - 0.4) / 0.6
        else:
            score = 1.0
        
        return float(max(0.0, min(score, 1.0)))