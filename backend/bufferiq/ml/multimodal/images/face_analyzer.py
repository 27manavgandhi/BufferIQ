"""Face detection and emotion analysis."""

from typing import List
import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.types import DetectedFace
from bufferiq.ml.multimodal.exceptions import AnalysisError


class FaceAnalyzer:
    """Analyze faces in images."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize face analyzer.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.emotions = ["happy", "neutral", "surprised", "sad", "angry"]
    
    def detect_faces(self, image: Image.Image) -> List[DetectedFace]:
        """
        Detect faces in image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected faces
            
        Raises:
            AnalysisError: If detection fails
        """
        try:
            # Simulate face detection
            # In production, use face_recognition, DeepFace, or similar
            detected = []
            
            # Heuristic: estimate if image likely contains faces
            # Based on aspect ratio and size
            width, height = image.size
            aspect_ratio = width / height
            
            # More likely to have faces if aspect ratio is reasonable
            if 0.5 < aspect_ratio < 2.0:
                np.random.seed(hash(str(image.size)) % 2**32)
                
                # Simulate 0-2 faces
                num_faces = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                
                for _ in range(num_faces):
                    face = DetectedFace(
                        bounding_box={
                            "x": float(np.random.random() * 0.5),
                            "y": float(np.random.random() * 0.4),
                            "width": float(0.2 + np.random.random() * 0.2),
                            "height": float(0.25 + np.random.random() * 0.25),
                        },
                        confidence=0.7 + np.random.random() * 0.25,  # 0.7-0.95
                        emotion=np.random.choice(self.emotions),
                        emotion_confidence=0.6 + np.random.random() * 0.3,  # 0.6-0.9
                    )
                    
                    if face.confidence >= self.confidence_threshold:
                        detected.append(face)
            
            return detected
            
        except Exception as e:
            raise AnalysisError(f"Face detection failed: {str(e)}")
    
    def analyze_emotions(self, faces: List[DetectedFace]) -> dict[str, float]:
        """
        Analyze emotion distribution in detected faces.
        
        Args:
            faces: List of detected faces
            
        Returns:
            Dictionary of emotion percentages
        """
        if not faces:
            return {}
        
        emotion_counts: dict[str, int] = {}
        for face in faces:
            if face.emotion:
                emotion_counts[face.emotion] = emotion_counts.get(face.emotion, 0) + 1
        
        total = len(faces)
        emotion_percentages = {
            emotion: count / total
            for emotion, count in emotion_counts.items()
        }
        
        return emotion_percentages