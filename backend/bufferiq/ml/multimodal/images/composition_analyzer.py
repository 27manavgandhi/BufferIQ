"""Image composition analysis."""

import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.types import CompositionScores
from bufferiq.ml.multimodal.exceptions import AnalysisError


class CompositionAnalyzer:
    """Analyze image composition quality."""
    
    def __init__(self):
        """Initialize composition analyzer."""
        pass
    
    def analyze(self, image: Image.Image) -> CompositionScores:
        """
        Analyze image composition.
        
        Args:
            image: PIL Image
            
        Returns:
            Composition scores
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Convert to grayscale for analysis
            gray = np.array(image.convert('L'), dtype=np.float32)
            
            # Calculate composition scores
            rule_of_thirds = self._score_rule_of_thirds(gray)
            golden_ratio = self._score_golden_ratio(gray)
            symmetry = self._score_symmetry(gray)
            balance = self._score_balance(gray)
            
            return CompositionScores(
                rule_of_thirds=rule_of_thirds,
                golden_ratio=golden_ratio,
                symmetry=symmetry,
                balance=balance,
            )
            
        except Exception as e:
            raise AnalysisError(f"Composition analysis failed: {str(e)}")
    
    def _score_rule_of_thirds(self, gray_image: np.ndarray) -> float:
        """
        Score adherence to rule of thirds.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Score (0-1)
        """
        height, width = gray_image.shape
        
        # Define third lines
        h_thirds = [height // 3, 2 * height // 3]
        w_thirds = [width // 3, 2 * width // 3]
        
        # Calculate edge density near third lines
        edge_density = 0.0
        margin = 20  # pixels
        
        for h in h_thirds:
            region = gray_image[max(0, h-margin):min(height, h+margin), :]
            edge_density += np.std(region)
        
        for w in w_thirds:
            region = gray_image[:, max(0, w-margin):min(width, w+margin)]
            edge_density += np.std(region)
        
        # Normalize
        score = min(edge_density / 1000.0, 1.0)
        return float(score)
    
    def _score_golden_ratio(self, gray_image: np.ndarray) -> float:
        """
        Score adherence to golden ratio.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Score (0-1)
        """
        height, width = gray_image.shape
        phi = 1.618
        
        # Define golden ratio lines
        h_golden = int(height / phi)
        w_golden = int(width / phi)
        
        # Calculate edge density near golden ratio lines
        edge_density = 0.0
        margin = 15
        
        for h in [h_golden, height - h_golden]:
            region = gray_image[max(0, h-margin):min(height, h+margin), :]
            edge_density += np.std(region)
        
        for w in [w_golden, width - w_golden]:
            region = gray_image[:, max(0, w-margin):min(width, w+margin)]
            edge_density += np.std(region)
        
        # Normalize
        score = min(edge_density / 800.0, 1.0)
        return float(score)
    
    def _score_symmetry(self, gray_image: np.ndarray) -> float:
        """
        Score image symmetry.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Score (0-1)
        """
        height, width = gray_image.shape
        
        # Vertical symmetry
        left = gray_image[:, :width//2]
        right = gray_image[:, width//2:]
        right_flipped = np.fliplr(right)
        
        # Resize to match if needed
        min_width = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_width]
        right_flipped = right_flipped[:, :min_width]
        
        v_symmetry = 1.0 - np.mean(np.abs(left - right_flipped)) / 255.0
        
        # Horizontal symmetry
        top = gray_image[:height//2, :]
        bottom = gray_image[height//2:, :]
        bottom_flipped = np.flipud(bottom)
        
        # Resize to match if needed
        min_height = min(top.shape[0], bottom_flipped.shape[0])
        top = top[:min_height, :]
        bottom_flipped = bottom_flipped[:min_height, :]
        
        h_symmetry = 1.0 - np.mean(np.abs(top - bottom_flipped)) / 255.0
        
        # Average symmetry
        score = (v_symmetry + h_symmetry) / 2.0
        return float(max(0.0, score))
    
    def _score_balance(self, gray_image: np.ndarray) -> float:
        """
        Score visual balance.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Score (0-1)
        """
        height, width = gray_image.shape
        
        # Calculate center of mass
        y_indices, x_indices = np.indices(gray_image.shape)
        total_mass = gray_image.sum()
        
        if total_mass == 0:
            return 0.5  # Neutral score for empty image
        
        center_y = (y_indices * gray_image).sum() / total_mass
        center_x = (x_indices * gray_image).sum() / total_mass
        
        # Calculate distance from image center
        image_center_y = height / 2
        image_center_x = width / 2
        
        distance = np.sqrt(
            ((center_y - image_center_y) / height) ** 2 +
            ((center_x - image_center_x) / width) ** 2
        )
        
        # Score: closer to center = better balance
        score = 1.0 - min(distance, 1.0)
        return float(score)