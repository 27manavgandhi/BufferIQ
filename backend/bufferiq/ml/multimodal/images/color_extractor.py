"""Color palette extraction from images."""

from typing import List, Tuple
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from bufferiq.ml.multimodal.types import ColorPalette
from bufferiq.ml.multimodal.exceptions import AnalysisError


class ColorExtractor:
    """Extract dominant colors from images."""
    
    def __init__(self, n_colors: int = 5):
        """
        Initialize color extractor.
        
        Args:
            n_colors: Number of dominant colors to extract
        """
        self.n_colors = n_colors
    
    def extract_palette(self, image: Image.Image) -> ColorPalette:
        """
        Extract color palette from image.
        
        Args:
            image: PIL Image
            
        Returns:
            Color palette
            
        Raises:
            AnalysisError: If extraction fails
        """
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for faster processing
            image = image.resize((150, 150), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Calculate percentages
            labels = kmeans.labels_
            percentages = [
                (labels == i).sum() / len(labels)
                for i in range(self.n_colors)
            ]
            
            # Sort by percentage
            sorted_indices = np.argsort(percentages)[::-1]
            dominant_colors = colors[sorted_indices].tolist()
            color_percentages = [percentages[i] for i in sorted_indices]
            
            return ColorPalette(
                dominant_colors=dominant_colors,
                color_percentages=color_percentages,
            )
            
        except Exception as e:
            raise AnalysisError(f"Color extraction failed: {str(e)}")
    
    def get_color_diversity(self, palette: ColorPalette) -> float:
        """
        Calculate color diversity score.
        
        Args:
            palette: Color palette
            
        Returns:
            Diversity score (0-1)
        """
        if len(palette.dominant_colors) < 2:
            return 0.0
        
        # Calculate average distance between colors
        colors = np.array(palette.dominant_colors)
        distances = []
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                dist = np.linalg.norm(colors[i] - colors[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 (max distance in RGB space is ~441)
        diversity = min(avg_distance / 441.0, 1.0)
        
        return float(diversity)