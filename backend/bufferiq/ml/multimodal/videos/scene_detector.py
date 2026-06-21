"""Scene detection in videos."""

from typing import List
import cv2
import numpy as np

from bufferiq.ml.multimodal.types import Scene
from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class SceneDetector:
    """Detect scenes in videos."""
    
    def __init__(self, threshold: float = 30.0):
        """
        Initialize scene detector.
        
        Args:
            threshold: Threshold for scene change detection
        """
        self.threshold = threshold
    
    def detect(self, video_path: str) -> List[Scene]:
        """
        Detect scenes in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of detected scenes
            
        Raises:
            MediaProcessingError: If detection fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise MediaProcessingError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            scenes = []
            
            prev_frame = None
            scene_start = 0.0
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    # Add final scene
                    if scenes or frame_idx > 0:
                        timestamp = frame_idx / fps if fps > 0 else 0
                        scenes.append(Scene(
                            start_time=scene_start,
                            end_time=timestamp,
                            duration=timestamp - scene_start,
                            scene_type="final",
                        ))
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate difference
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = np.mean(diff)
                    
                    # Scene change detected
                    if diff_score > self.threshold:
                        timestamp = frame_idx / fps if fps > 0 else 0
                        
                        # Close previous scene
                        scenes.append(Scene(
                            start_time=scene_start,
                            end_time=timestamp,
                            duration=timestamp - scene_start,
                            scene_type="transition",
                        ))
                        
                        scene_start = timestamp
                
                prev_frame = gray
                frame_idx += 1
            
            cap.release()
            
            return scenes
            
        except Exception as e:
            raise MediaProcessingError(f"Scene detection failed: {str(e)}")