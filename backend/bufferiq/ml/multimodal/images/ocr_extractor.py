"""OCR text extraction from images."""

from typing import List
import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.types import ExtractedText
from bufferiq.ml.multimodal.exceptions import AnalysisError


class OCRExtractor:
    """Extract text from images using OCR."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize OCR extractor.
        
        Args:
            confidence_threshold: Minimum confidence for text extraction
        """
        self.confidence_threshold = confidence_threshold
        # In production, use pytesseract or similar
    
    def extract_text(self, image: Image.Image) -> List[ExtractedText]:
        """
        Extract text from image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of extracted text
            
        Raises:
            AnalysisError: If extraction fails
        """
        try:
            # Simulate OCR
            # In production, use pytesseract or EasyOCR
            extracted = []
            
            # Check if image is likely to contain text (heuristic)
            img_array = np.array(image.convert('L'))  # Grayscale
            edge_density = self._estimate_edge_density(img_array)
            
            if edge_density > 0.1:  # Likely contains text
                # Simulate finding text
                sample_texts = [
                    "Sample Text",
                    "Important Message",
                    "Click Here",
                ]
                
                np.random.seed(hash(str(image.size)) % 2**32)
                num_texts = min(2, len(sample_texts))
                
                for i in range(num_texts):
                    text = ExtractedText(
                        text=sample_texts[i],
                        confidence=0.6 + np.random.random() * 0.3,  # 0.6-0.9
                        position={
                            "x": float(np.random.random() * 0.6),
                            "y": float(np.random.random() * 0.6),
                            "width": float(0.2 + np.random.random() * 0.2),
                            "height": float(0.05 + np.random.random() * 0.05),
                        }
                    )
                    
                    if text.confidence >= self.confidence_threshold:
                        extracted.append(text)
            
            return extracted
            
        except Exception as e:
            raise AnalysisError(f"Text extraction failed: {str(e)}")
    
    def _estimate_edge_density(self, gray_image: np.ndarray) -> float:
        """
        Estimate edge density in image.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Edge density score (0-1)
        """
        # Simple edge detection using gradient
        dy = np.abs(np.diff(gray_image, axis=0))
        dx = np.abs(np.diff(gray_image, axis=1))
        
        edge_score = (dy.mean() + dx.mean()) / 255.0
        return float(edge_score)